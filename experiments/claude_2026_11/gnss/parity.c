/* 对拍：读 20 Hz 的 ax ay az（mg，一行三个整数），每秒输出 loco 与原因 */
#include <stdio.h>
#include "gnss_trigger.h"
int main(void) {
    gt_state_t s; gt_init(&s, GT_PRESET_STANDARD);
    int x, y, z, k = 0;
    while (scanf("%d %d %d", &x, &y, &z) == 3) {
        gt_feed(&s, (int16_t)x, (int16_t)y, (int16_t)z);
        if (++k % 20 == 0) { gt_reason_t r = gt_tick(&s); unsigned char lo = s.loco_ring[(s.loco_i + 14) % 15]; printf("%d %d\n", lo, (int)r); }
    }
    return 0;
}
