/* 院子场景：狗在院子里溜达 30 分钟，蓝牙每隔一两分钟断开 40~90 秒；之后真出门走远。
   比较：只防抖 30 秒（关掉冷却和围栏）/ 新方案。 */
#include <stdio.h>
#include "gnss_trigger.h"
static int wave(int k, int on) { if (!on) return 1000; int ph = k % 9; return 1000 + (ph < 3 ? 150 : ph < 5 ? -80 : 0); } /* 约 2.2 步/秒 */
static void run(int use_new) {
    gt_state_t s; gt_init(&s, GT_PRESET_SAVE); gt_set_link(&s, 1);
    int k = 0, yard_fix = 0, first_out = -1, link = 1;
    double x = 0;                                 /* 离家距离（米），院子里在 15~35 m 之间晃 */
    for (int sec = 0; sec < 2700; sec++) {
        int walking, want_link;
        if (sec < 1800) {                         /* 院子 30 分钟：走 40 秒、歇 40 秒 */
            walking = (sec / 40) % 2 == 0;
            x = 25 + 10 * ((sec / 70) % 2 ? 1 : -1) * ((sec % 70) / 70.0);
            want_link = !((sec / 75) % 2 == 1 && (sec % 75) < 60);   /* 每 150 秒断 60 秒左右 */
        } else {                                  /* 出门：1.3 m/s 一直往外走 */
            walking = 1; x = 35 + (sec - 1800) * 1.3; want_link = 0;
        }
        if (want_link != link) { gt_set_link(&s, (uint8_t)want_link); link = want_link; }
        for (int i = 0; i < 20; i++, k++) gt_feed(&s, 0, 0, (int16_t)wave(k, walking));
        gt_reason_t r = gt_tick(&s);
        if (r != GT_NONE) {
            if (sec < 1800) yard_fix++;
            else if (first_out < 0) first_out = sec;
            if (use_new) gt_on_fix_home(&s, 10, (uint16_t)x);
        }
        if (!use_new) { s.left_once = 0; s.yard = 0; }   /* 关掉冷却和院子，等于老办法 */
    }
    printf("%s：院子 30 分钟打了 %d 个点；出门后第一个点在出门 %d 秒后（约 %.0f 米）\n",
           use_new ? "新方案" : "只防抖 30 秒", yard_fix, first_out - 1800, (first_out - 1800) * 1.3);
}
int main(void) { run(0); run(1); return 0; }
