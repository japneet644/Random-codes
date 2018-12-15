#include<iostream>

using namespace std;
int modulo(int a,int b, int c)
{
  if(b==1) return a%c;
  int ans = modulo(a,b/2,c);
  if(b%2==0) return (ans*ans)%c;
   return (a*ans*ans)%c;
}
int main(int argc, char const *argv[]) {
  if(5>3-1)  std::cout << "ajbajma " << '\n';
}
