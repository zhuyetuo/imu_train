/* IMU 智能打点参考实现。说明见 gnss_trigger.h。只用整数运算，没有浮点、没有 malloc。 */
#include "gnss_trigger.h"
#include <string.h>

#define HZ              20
#define PEAK_MG         60    /* 去基线后高于 60 mg 的局部最大值算一步 */
#define MIN_GAP         5     /* 两步至少隔 0.25 s（抓挠 5–8 Hz 的峰挤在一起，大部分被这条吃掉） */
#define MIN_STEPS       5     /* 4 秒里至少 5 步 */
#define MAX_STEPS       18    /* 4 秒里最多 18 步（步频 ≤4.5/s） */
#define MAX_CV_PCT      45    /* 步间隔变异系数 ≤45%：节奏要均匀 */
#define SUS_WIN         15    /* 持续移动：15 秒里… */
#define SUS_MIN         10    /* …至少 10 秒在走 / 跑 */
#define REST_BEFORE_DEP 60    /* 静止 60 秒后再动起来才算「出发」 */
#define STOP_S          60    /* 停下 60 秒算到达 */
#define STEP_CM         60    /* 每步估 60 cm（待用真实 GNSS 标定） */

static const gt_preset_t PRESETS[3] = {
    /* 省电（默认）：只想看大概去了哪 */
    { 20, 100, 300, 20, 4u * 3600u },
    /* 标准 */
    { 10, 50, 120, 10, 2u * 3600u },
    /* 精细：想看在小公园里怎么跑——走 15 米就打一个、最长 30 秒一个 */
    { 0, 15, 30, 5, 3600u },
};

static uint32_t isqrt32(uint32_t v) {
    uint32_t r = 0, b = 1u << 30;
    while (b > v) b >>= 2;
    while (b) {
        if (v >= r + b) { v -= r + b; r = (r >> 1) + b; } else r >>= 1;
        b >>= 2;
    }
    return r;
}

void gt_set_custom(gt_state_t *s, const gt_preset_t *p) { s->p = *p; }
void gt_set_preset(gt_state_t *s, gt_preset_id_t id) { s->p = PRESETS[id <= GT_PRESET_FINE ? id : 0]; }

void gt_init(gt_state_t *s, gt_preset_id_t preset) {
    memset(s, 0, sizeof(*s));
    gt_set_preset(s, preset);
    s->base = 1000 << 8;      /* 1 g */
    s->still_s = 0xFFFFFFFFu / 2;
    s->dist_scale = 1;
    s->armed = 1;             /* 没报过蓝牙状态：按「没连着」工作（老用法兼容） */
    s->leave_age_s = 0xFFFFFFF0u;
    s->yard_scale = 1;
}

static void reset_track(gt_state_t *s) {
    s->moving = s->pending_depart = s->departed = 0;
    s->stop_s = 0; s->dist_cm = s->depart_cm = 0;
    s->still_s = 0; s->since_fix_s = 0;
    s->dist_scale = 1;
}

void gt_set_link(gt_state_t *s, uint8_t connected) {
    if (connected) {
        if (!s->link) s->link_s = 0;
        s->link = 1; s->armed = 0; s->unlink_s = 0;   /* 连上：立刻停，位置交给网关 / 手机 */
        reset_track(s);
    } else if (s->link) {
        s->link = 0; s->unlink_s = 0;                 /* 断开：先等满防抖时间 */
        s->unlink_walk_cm = 0;
    }
}

/* 每个样本：|a| 去掉 ~1 s 的慢基线，两点平均，找局部最大值记一步 */
void gt_feed(gt_state_t *s, int16_t ax, int16_t ay, int16_t az) {
    uint32_t m = isqrt32((uint32_t)((int32_t)ax * ax + (int32_t)ay * ay + (int32_t)az * az));
    s->base += (((int32_t)m << 8) - s->base) / HZ;          /* EMA，时间常数约 1 s */
    int32_t d = (int32_t)m - (s->base >> 8);
    int32_t cur = (s->draw + d) / 2;                         /* 跟上一个原始样本两点平均，去毛刺 */
    s->draw = d;
    s->t++;
    uint32_t pt = s->t - 1;                                  /* prev1 所在的样本 */
    /* prev1 是上一个平滑值：比前后都大、超过门限、离上一步够远 → 一步 */
    if (s->prev1 > PEAK_MG && s->prev1 >= s->prev2 && s->prev1 > cur &&
        (s->step_cnt == 0 || pt - s->last_step_t >= MIN_GAP)) {
        s->last_step_t = pt;
        s->step_t[s->step_head] = pt;
        s->step_head = (uint8_t)((s->step_head + 1) % 24);
        if (s->step_cnt < 24) s->step_cnt++;
        if (s->steps_this_sec < 255) s->steps_this_sec++;
    }
    s->prev2 = s->prev1;
    s->prev1 = cur;
}

/* 最近 4 秒：步数够、步频不离谱、步间隔均匀（变异系数 ≤ MAX_CV） → 在走 / 跑 */
static uint8_t loco_now(gt_state_t *s) {
    uint32_t from = s->t > 4u * HZ ? s->t - 4u * HZ : 0;
    uint32_t ts[24]; uint8_t n = 0;
    for (uint8_t k = 0; k < s->step_cnt; k++) {               /* 从旧到新取窗口里的步 */
        uint8_t i = (uint8_t)((s->step_head + 24 - s->step_cnt + k) % 24);
        if (s->step_t[i] >= from) ts[n++] = s->step_t[i];
    }
    if (n < MIN_STEPS || n > MAX_STEPS) return 0;
    int64_t sum = 0, sq = 0;
    for (uint8_t k = 1; k < n; k++) { int64_t iv = (int64_t)(ts[k] - ts[k - 1]); sum += iv; sq += iv * iv; }
    int64_t m = n - 1;
    /* 方差 / 均值² ≤ CV²  ⇔  (m·sq − sum²)·100² ≤ CV%²·sum² */
    return (m * sq - sum * sum) * 10000 <= (int64_t)MAX_CV_PCT * MAX_CV_PCT * sum * sum;
}

gt_reason_t gt_tick(gt_state_t *s) {
    /* 1) 这一秒是不是在走 / 跑，最近 15 秒里有几秒 */
    uint8_t loco = loco_now(s);
    uint32_t step_cm = (uint32_t)s->steps_this_sec * STEP_CM;
    s->steps_this_sec = 0;
    s->loco_ring[s->loco_i] = loco;
    s->loco_i = (uint8_t)((s->loco_i + 1) % SUS_WIN);
    uint8_t cnt = 0;
    for (int i = 0; i < SUS_WIN; i++) cnt += s->loco_ring[i];
    uint8_t sus = cnt >= SUS_MIN;
    if (s->leave_age_s < 0xFFFFFFF0u) s->leave_age_s++;
    if (sus) s->leave_path_cm += step_cm;      /* 蓝牙连着也算：院子里进进出出走了多少 */

    /* 2) 蓝牙开关：连着就不工作（计步照常跑，状态保持热的）；断开满 30 秒打「离开点」再开始 */
    if (s->link) {
        if (s->link_s < 0xFFFFFFF0u) s->link_s++;
        if (s->link_s >= GT_YARD_EXIT_S) { s->yard = 0; s->yard_scale = 1; }   /* 回屋待够了，退出院子 */
        return GT_NONE;
    }
    if (!s->armed) {
        if (s->unlink_s < 0xFFFF) s->unlink_s++;
        if (sus) s->unlink_walk_cm += step_cm;
        if (s->yard) {
            /* 院子里：断一会儿又连上很正常，不打离开点；一直断着还走了不少路，才打一个看看出没出院子 */
            if (s->unlink_s < GT_YARD_OUT_S ||
                s->unlink_walk_cm < (uint32_t)GT_YARD_STEP_M * s->yard_scale * 100u) return GT_NONE;
            s->armed = 1; reset_track(s);
            s->has_fix = 1; s->moving = sus; s->departed = 1;
            s->last_reason = GT_LEAVE;
            return GT_LEAVE;
        }
        if (s->unlink_s < GT_LINK_DEBOUNCE_S) return GT_NONE;
        s->armed = 1;
        reset_track(s);
        s->has_fix = 1;
        s->moving = sus;                 /* 已经在走：接着按「移动中」打点，不再等出发 */
        s->departed = 1;                 /* 离开就算出发过：之后停下来照样打「到达」点 */
        /* 离开点冷却：刚打过离开点、之后也没走多远——多半是院子里蓝牙忽断忽连，悄悄接着工作，不再打 */
        if (s->left_once && s->leave_age_s < GT_LEAVE_COOLDOWN_S &&
            s->leave_path_cm < (uint32_t)GT_LEAVE_MIN_M * 100u) {
            s->last_reason = GT_NONE;
            return GT_NONE;
        }
        s->left_once = 1; s->leave_age_s = 0; s->leave_path_cm = 0;
        s->last_reason = GT_LEAVE;
        return GT_LEAVE;
    }
    if (s->has_fix) s->since_fix_s++;


    /* 3) 打点状态机 */
    if (sus) {
        s->dist_cm += step_cm;
        if (!s->moving) {
            s->moving = 1; s->stop_s = 0; s->departed = 0;
            s->pending_depart = s->still_s >= REST_BEFORE_DEP;
            s->depart_cm = 0;
        }
        s->still_s = 0;
        if (s->pending_depart) {
            s->depart_cm += step_cm;
            if (s->depart_cm >= (uint32_t)s->p.depart_min_m * 100u) {
                s->pending_depart = 0; s->departed = 1;
                s->dist_cm = 0; s->since_fix_s = 0; s->has_fix = 1;
                s->dist_scale = 1;           /* 新的一趟：上一处转悠放大的间隔清掉 */
                s->last_reason = GT_DEPART;
                return GT_DEPART;
            }
        } else if (s->dist_cm >= (uint32_t)s->p.move_dist_m * s->dist_scale * 100u ||
                   (s->has_fix && s->since_fix_s >= s->p.move_max_s)) {
            s->dist_cm = 0; s->since_fix_s = 0; s->has_fix = 1;
            s->last_reason = GT_MOVING;
            return GT_MOVING;
        }
        return GT_NONE;
    }
    s->still_s++;
    if (s->moving && ++s->stop_s >= STOP_S) {
        s->moving = 0; s->pending_depart = 0;
        if (s->departed && s->dist_cm >= (uint32_t)s->p.arrive_min_m * 100u) {
            s->dist_cm = 0; s->since_fix_s = 0; s->has_fix = 1;
            s->last_reason = GT_ARRIVE;
            return GT_ARRIVE;
        }
    }
    if (!s->moving && (!s->has_fix || s->since_fix_s >= s->p.heartbeat_s)) {
        s->since_fix_s = 0; s->has_fix = 1;
        s->last_reason = GT_HEARTBEAT;
        return GT_HEARTBEAT;
    }
    return GT_NONE;
}

/* 打完一个点，告诉它跟上一个点实际差多远：
 * 「移动中」的点实际只挪了不到 move_dist 的三分之一 → 多半是在一小块地方来回转，把间隔放大一倍（最多 4 倍）；
 * 实际挪得够远就恢复。不调这个函数也能用，只是徘徊时会多打点。 */
void gt_on_fix(gt_state_t *s, uint16_t moved_m) {
    if (s->last_reason != GT_MOVING) return;   /* 出发 / 到达 / 保底 / 离开点不拿来判断转悠 */
    uint32_t want = (uint32_t)s->p.move_dist_m * s->dist_scale;
    if (moved_m * 3u < want) { if (s->dist_scale < 4) s->dist_scale <<= 1; }
    else if (moved_m * 3u >= want * 2u) s->dist_scale = 1;
}

/* 打完点，再告诉它离家多远：50 m 以内进「院子」状态（院子里只按走过的距离打点），出了围栏恢复正常 */
void gt_on_fix_home(gt_state_t *s, uint16_t moved_m, uint16_t home_m) {
    gt_on_fix(s, moved_m);
    if (home_m <= GT_YARD_R_M) {
        /* 还在院子里：回到「等」的状态，下次要连续断开更久、走更远才打（翻倍，最多 8 倍） */
        if (s->yard && s->yard_scale < 8) s->yard_scale <<= 1;
        s->yard = 1;
        if (!s->link) { s->armed = 0; s->unlink_s = 0; s->unlink_walk_cm = 0; }
    } else {
        s->yard = 0; s->yard_scale = 1;           /* 出了院子：照常打点 */
    }
}
