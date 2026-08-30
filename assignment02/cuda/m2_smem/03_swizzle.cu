#include <cstring>

// TODO: 实现三个映射。
static int swizzle_128B(int row, int colByte) { 
    int low=colByte&0xf;
    int chunk=colByte>>4;
    int phys=chunk^(row&0x7);
    return row*128+(phys<<4)+low;
 }
static int swizzle_64B(int row, int colByte) { 
    int low=colByte&0xf;
    int chunk=colByte>>4;
    int phys=chunk^(row&0x3);
    return row*64+(phys<<4)+low;
 }
static int swizzle_32B(int row, int colByte) { 
    int low=colByte&0xf;
    int chunk=colByte>>4;
    int phys=chunk^(row&0x1);
    return row*32+(phys<<4)+low;
 }

// 以下为判测,不需要修改。
static int check_mode(const char* name, int (*fn)(int, int), int rowBytes,
                      int period) {
    const int rows = 8;
    int atom = rows * rowBytes;
    static char hit[8 * 128];
    memset(hit, 0, atom);
    int bad = 0;
    // 约定:输出是 atom 内偏移,行基址 row*rowBytes 由映射自己含入。
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < rowBytes; c++) {
            int off = fn(r, c);
            if (off < 0 || off >= atom || hit[off]) bad++;
            else hit[off] = 1;
        }
    if (bad) {
        printf("%s FAIL:双射检查,%d 处越界或碰撞\n", name, bad);
        return 1;
    }
    // 列访问:固定 16B chunk j,一个周期内的行的物理 chunk 应两两不同
    int chunks = rowBytes / 16;
    for (int j = 0; j < chunks; j++) {
        unsigned seen = 0;
        for (int r = 0; r < period; r++) {
            int off = fn(r, j * 16);
            int physChunk = (off % rowBytes) / 16;
            if (seen & (1u << physChunk)) {
                printf("%s FAIL:列 %d 在行 0..%d 内物理 chunk 重复\n", name,
                       j, period - 1);
                return 1;
            }
            seen |= 1u << physChunk;
        }
    }
    printf("%s PASS\n", name);
    return 0;
}

int main() {
    int bad = 0;
    bad += check_mode("128B", swizzle_128B, 128, 8);
    bad += check_mode(" 64B", swizzle_64B, 64, 4);
    bad += check_mode(" 32B", swizzle_32B, 32, 2);
    return bad != 0;
}
