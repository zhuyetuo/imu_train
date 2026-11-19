/* 蓝牙开关场景测试：连着时不打点；断开 30 秒打离开点；再连上立刻停 */
#include <stdio.h>
#include <math.h>
#include "gnss_trigger.h"
static int walk_sample(int k) { /* 2.2 步/秒的走路：|a| 上每步一个 ~150 mg 的峰 */
    double t = k / 20.0; return (int)(1000 + 150 * sin(2 * 3.14159265 * 2.2 * t));
}
int main(void) {
    gt_state_t s; gt_init(&s, GT_PRESET_SAVE);
    gt_set_link(&s, 1);
    int k = 0, fixes_connected = 0, leave_at = -1, after = 0;
    for (int sec = 0; sec < 600; sec++) {
        if (sec == 120) gt_set_link(&s, 0);       /* 2 分钟时出门，蓝牙断开 */
        if (sec == 500) gt_set_link(&s, 1);       /* 回家连上 */
        int walking = sec < 400;                  /* 400 秒后停下来歇着 */
        for (int i = 0; i < 20; i++, k++) gt_feed(&s, 0, 0, (int16_t)(walking ? walk_sample(k) : 1000));
        gt_reason_t r = gt_tick(&s);
        if (r != GT_NONE) {
            if (sec < 120 || sec >= 500) fixes_connected++;
            if (r == GT_LEAVE) leave_at = sec;
            else if (sec < 500) after++;
            printf("%3d s: 打点 %d\n", sec, (int)r);
        }
    }
    printf("连着时打点 %d 个（应为 0）；离开点在第 %d 秒（应为 150 左右）；之后走路中打点 %d 个\n", fixes_connected, leave_at, after);
    return 0;
}
