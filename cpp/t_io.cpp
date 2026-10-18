#include <bits/stdc++.h>
using namespace std;
int main(){
    char buf[1024];
    cout << "getcwd: " << (getcwd(buf,1024)? buf : "FAIL") << endl;
    ifstream f("all_corpus.bin", ios::binary|ios::ate);
    cout << "open all_corpus.bin: " << (f? "OK" : "FAIL") << " size=" << (f?(long long)f.tellg():-1) << endl;
    ifstream g2("hongloumeng.txt", ios::binary|ios::ate);
    cout << "open hongloumeng.txt: " << (g2? "OK" : "FAIL") << endl;
    return 0;
}
