#include<iostream>
using namespace std;

// template<typename T>
class use
{
public:
  int x; float y;
  use(const int s,const float value): x(s), y(value){}
  use(): x(0), y(1.0){}
  use& operator=(const use& A){x = (int)A.y; y = (float)A.x;}
};

int main(int argc, char const *argv[]) {
  use p(1,2);
  use s;
  cout<<p.x<<s.x<<'\n';
  s = p;
  cout<<s.x<<s.y<<'\n';
  return 0;
}
