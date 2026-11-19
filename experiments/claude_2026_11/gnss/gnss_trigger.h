/*
 * IMU 智能打点：只用 3 轴加速度计，决定什么时候调用卫星定位（GNSS）打一个点。
 *
 * 适用场景：狗独自外出，连不上家里网关、也连不上主人手机（两者都是蓝牙），只能靠卫星定位。
 * 思路：由蓝牙状态开关——
 *   蓝牙连着（网关 / 手机在）：智能打点停着，卫星关着，位置交给网关 / 手机；
 *   蓝牙断开满 30 秒（防门口信号时有时无）：打一个「离开点」，智能打点开始工作；
 *   重新连上：立刻停。
 * 院子（蓝牙边缘，忽断忽连）：院子里是「断一会儿又连上」，真出门是「一直断着、还在走」——
 *   院子围栏：App 下发家的坐标，打完点用 gt_on_fix_home 告诉它离家多远，50 m 以内算「在院子」；
 *   在院子里蓝牙断开不打离开点；要「连续断开满 90 秒、且这段时间估算走了 ≥50 m」才打一个点看看出没出院子，
 *   那个点还在院子里就把要求的距离翻倍（50 → 100 → 200 → 400 m），出了围栏恢复正常；
 *   回屋里蓝牙连续连上 10 分钟，退出院子状态。
 *   离开点冷却（不知道家在哪时的兜底）：打过离开点 10 分钟内又断开、期间也没走够 50 m，就不再打离开点。
 * 工作时 GNSS 平时关着；加速度计看到「持续在走 / 跑」才打点——出发、移动中按估算距离、到达，
 * 静止很久再补一个保底点。原地休息、来回走几步都不打。
 *
 * 用法（20 Hz）：
 *   gt_init(&st, GT_PRESET_SAVE);          // 默认省电档；用户在 App 里切档就调 gt_set_preset
 *   gt_set_link(&st, ble_connected());     // 开机先报一次蓝牙状态；之后每次连上 / 断开的通知都调
 *   每个加速度样本：  gt_feed(&st, ax, ay, az);     // 单位 mg（int16），蓝牙连着也照常喂（计步状态保持热的）
 *   每秒一次：        r = gt_tick(&st);             // r != GT_NONE 就去打一个点
 *   打完点（可选）：  gt_on_fix_home(&st, moved_m, home_m);  // 跟上一个点差多远、离家多远（不知道家在哪就用 gt_on_fix）
 *
 * 参数来源：Vehkaoja 2022 公开数据（颈部项圈）验证计步 / 移动检测，狗场 36 个狗·天真实数据模拟打点。
 * 每步多少米（step_cm）还没用真实 GNSS 标定，先按 60 cm。
 */
#ifndef GNSS_TRIGGER_H
#define GNSS_TRIGGER_H
#include <stdint.h>

/* GT_LEAVE：蓝牙断开满 30 秒，刚离开网关 / 手机范围，打第一个点 */
typedef enum { GT_NONE = 0, GT_DEPART, GT_MOVING, GT_ARRIVE, GT_HEARTBEAT, GT_LEAVE } gt_reason_t;

/* 打点密度三档（用户可调，默认省电）。也可以直接填 gt_preset_t 自定义 */
typedef enum { GT_PRESET_SAVE = 0, GT_PRESET_STANDARD = 1, GT_PRESET_FINE = 2 } gt_preset_id_t;

typedef struct {
    uint16_t depart_min_m;   /* 动起来以后估算走出多远才打「出发」点（笼子里起身走两步不打） */
    uint16_t move_dist_m;    /* 移动中每走这么远打一个点 */
    uint16_t move_max_s;     /* 移动中离上个点最长多久必打一个 */
    uint16_t arrive_min_m;   /* 停下以后，上个点之后走了这么远才打「到达」点 */
    uint32_t heartbeat_s;    /* 一直不动：隔这么久打一个保底点 */
} gt_preset_t;

typedef struct {
    /* 计步 */
    int32_t base;            /* |a| 的慢基线（Q8，mg） */
    int32_t prev2, prev1;    /* 平滑后的最近两个样本（mg） */
    int32_t draw;            /* 上一个去基线、没平滑的样本（mg） */
    uint32_t t;              /* 样本计数 */
    uint32_t last_step_t;    /* 上一步在第几个样本 */
    uint32_t step_t[24];     /* 最近 24 步的样本时刻（环形），算最近 4 秒的步数和节奏 */
    uint8_t step_head, step_cnt;
    uint8_t steps_this_sec;
    uint8_t loco_ring[15];   /* 最近 15 秒每秒是否在走 / 跑 */
    uint8_t loco_i;
    /* 打点状态 */
    gt_preset_t p;
    uint8_t moving, pending_depart, departed;
    uint32_t still_s, stop_s, since_fix_s;
    uint32_t dist_cm, depart_cm;
    uint8_t has_fix;
    uint8_t dist_scale;      /* 原地徘徊时把 move_dist 放大（1~4 倍），见 gt_on_fix */
    uint8_t last_reason;     /* 上一次 gt_tick 让打的是哪种点（gt_on_fix 只看「移动中」的点） */
    /* 蓝牙开关 */
    uint8_t link;            /* 1 = 蓝牙连着（网关 / 手机在），智能打点停着 */
    uint8_t armed;           /* 1 = 智能打点在工作 */
    uint16_t unlink_s;       /* 断开了多少秒（满 GT_LINK_DEBOUNCE_S 才开始工作） */
    /* 院子（蓝牙边缘忽断忽连） */
    uint8_t left_once;       /* 打过离开点没有 */
    uint32_t leave_age_s;    /* 离上一个离开点多少秒 */
    uint32_t leave_path_cm;  /* 上一个离开点之后估算走了多远（蓝牙连着时也算） */
    uint8_t yard;            /* 1 = 最近一个点在家附近（院子里） */
    uint8_t yard_scale;      /* 院子里打点要求的距离倍数（1~8，点还在院子里就翻倍） */
    uint32_t unlink_walk_cm; /* 这一次连续断开以来估算走了多远 */
    uint32_t link_s;         /* 这一次连续连着多少秒 */
} gt_state_t;

#define GT_LINK_DEBOUNCE_S 30     /* 蓝牙断开满这么久才算真断 */
#define GT_LEAVE_COOLDOWN_S 600   /* 离开点冷却：10 分钟 */
#define GT_LEAVE_MIN_M 50         /* 冷却期内走够这么远才再打离开点 */
#define GT_YARD_R_M 50            /* 离家这么近算院子 */
#define GT_YARD_STEP_M 50         /* 院子里：连续断开时估算走够这么远（× 倍数）才打一个点 */
#define GT_YARD_OUT_S 90          /* 院子里：连续断开满这么久才可能是出院子 */
#define GT_YARD_EXIT_S 600        /* 蓝牙连续连上这么久：回屋了，退出院子状态 */

void gt_init(gt_state_t *s, gt_preset_id_t preset);
void gt_set_preset(gt_state_t *s, gt_preset_id_t preset);
void gt_set_custom(gt_state_t *s, const gt_preset_t *p);
void gt_feed(gt_state_t *s, int16_t ax_mg, int16_t ay_mg, int16_t az_mg);
gt_reason_t gt_tick(gt_state_t *s);
void gt_on_fix(gt_state_t *s, uint16_t moved_m);
/* 蓝牙连上 / 断开的通知：connected=1 连上（停），0 断开（30 秒后开始工作）。不调的话默认按「没连着」一直工作 */
void gt_set_link(gt_state_t *s, uint8_t connected);
/* 打完点：moved_m 跟上一个点差多远，home_m 离家多远（App 下发家的坐标后才有）。50 m 内进院子状态，出了就恢复 */
void gt_on_fix_home(gt_state_t *s, uint16_t moved_m, uint16_t home_m);

#endif
