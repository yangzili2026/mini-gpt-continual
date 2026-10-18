// ============================================================
// C++ 造大语言模型 · 第1课扩展：中文语料试验（《红楼梦》）
// ------------------------------------------------------------
// 第1课的词表是全部 256 个字节。中文在 UTF-8 编码里每个字 = 3 个字节，
// 所以词表天然支持中文，本程序验证这一点，并统计《红楼梦》里
// 出现最多的汉字是哪些。
// ============================================================
// 标注说明：【难】=没学过重点看注释 【易错】=容易写错 【记】=照抄即可
// ============================================================
#include <bits/stdc++.h>   // 万能头
using namespace std;

int main(int argc, char* argv[]) {
    // ---------- 1. 读文件（和第1课一样） ----------
    // 【难】main 可以带参数：argc = 参数的个数，argv = 参数字符串数组。
    //   运行时敲 ./lesson1b_chinese 红楼梦.txt，argv[1] 就是"红楼梦.txt"。
    //   不带参数时默认用《红楼梦》。
    string filename = (argc >= 2) ? argv[1] : "hongloumeng.txt";

    ifstream fin(filename, ios::binary);   // 【易错】ios::binary 不能少
    if (!fin) {
        cout << "找不到文件：" << filename << endl;
        return 1;
    }
    vector<unsigned char> data(
        (istreambuf_iterator<char>(fin)),
        istreambuf_iterator<char>());
    fin.close();
    cout << "读入 " << data.size() << " 个字节（" << filename << "）" << endl;

    // ---------- 2. 先看字节排行榜（和第1课相同思路） ----------
    vector<long long> byteCount(256, 0);
    for (unsigned char c : data) byteCount[c]++;

    vector<int> order(256);
    for (int i = 0; i < 256; i++) order[i] = i;
    sort(order.begin(), order.end(),
         [&](int a, int b){ return byteCount[a] > byteCount[b]; });

    cout << "字节排行榜 Top 8（注意：排在最前的都是 200 多的大数字）" << endl;
    for (int k = 0; k < 8; k++) {
        int c = order[k];
        printf("  第%2d名: 字节 %3d  出现了 %lld 次，占 %.2f%%\n",
               k + 1, c, byteCount[c], 100.0 * byteCount[c] / data.size());
    }

    // ---------- 3. 【难】把 UTF-8 的中文"拼装"出来再统计 ----------
    // UTF-8 规则（只需记住一条）：
    //   字节 <  128        -> 英文字母/符号，占 1 个字节
    //   字节 >= 224 (0xE0) -> 汉字的开头，后面还跟着 2 个字节，3 个字节 = 1 个汉字
    // 例如"红" = 231 191 205 三个字节，缺一个就变成乱码。
    // 我们把连续 3 个字节拼成一个整数（叫"码点"），用它当作汉字的编号来计数。
    map<int, long long> charCount;   // 键 = 汉字的码点，值 = 出现次数
    long long chineseChars = 0;      // 全书汉字总数

    for (size_t i = 0; i < data.size(); ) {
        unsigned char b = data[i];
        if (b < 128) {
            i++;                          // 英文字母，跳过 1 个字节
        } else if (b >= 0xE0 && i + 2 < data.size()) {
            // 【易错】左移 16 位 / 8 位，把 3 个字节拼进一个 int
            int code = (b << 16) | (data[i+1] << 8) | data[i+2];
            charCount[code]++;
            chineseChars++;
            i += 3;                       // 汉字占 3 个字节，一次跳 3 格
        } else {
            i++;                          // 其他情况（少见），安全跳 1 格
        }
    }
    cout << "全书汉字约 " << chineseChars << " 个" << endl;

    // ---------- 4. 汉字排行榜 Top 15 ----------
    // map 默认按码点排，我们要按次数排，所以倒进 vector 再 sort
    vector<pair<int, long long>> chars(charCount.begin(), charCount.end());
    sort(chars.begin(), chars.end(),
         [](const pair<int, long long>& a, const pair<int, long long>& b){
             return a.second > b.second;   // 【易错】按"次数"(second)从大到小
         });

    cout << "汉字排行榜 Top 15：" << endl;
    for (int k = 0; k < 15 && k < (int)chars.size(); k++) {
        int code = chars[k].first;
        // 【记】把码点拆回 3 个字节，放进字符串，cout 就能原样显示汉字
        string s;
        s.push_back((char)((code >> 16) & 255));
        s.push_back((char)((code >> 8) & 255));
        s.push_back((char)(code & 255));
        cout << "  第" << setw(2) << k + 1 << "名: '" << s << "'  出现了 "
             << chars[k].second << " 次" << endl;
    }
    return 0;
}
