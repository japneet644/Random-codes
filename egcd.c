#include <stdio.h>

void egcd(int a, int b)
{
    int q,xc = 0,xpr =1,yc = 1,ypr=0;
    while(b!=0)
    {
        printf("%d %d\n",a,b);
        q = a/b;
        int x = xpr - q*xc;
        int y = ypr - q*yc;
        ypr = yc; xpr = xc;
        yc = y; xc= x;
        q = a;
        a = b;
        b = q%b;
    }
    printf("%d %d\n",xpr,ypr);
}

int main(int argc, char const *argv[]) {
    egcd(1914,899);
    return 0;
}
