#include<stdio.h>
#include<string.h>

int mod(char b[],int a,int n);
int gcd(int a,int b);

int main()
{
    int t,a,i,m,n,res;
    char b[500],tem;
    scanf("%d",&t);
    while(t--)
    {
              scanf("%d",&a);
              scanf("%s",b);
              n=strlen(b);
              if(a==0)
                      printf("%s\n",b);
              else
              {
                  m=mod(b,a,n);
                  res=gcd(a,m);
                  printf("%d\n",res);
              }
    }
    scanf("%c%c",&tem,&tem);
    return 0;
}

int gcd(int a,int b)
{
	if(b==0)
		return a;
	else
		return gcd(b,a%b);
}

int mod(char b[],int a,int n)
{
    if(n==1){return (b[0]-'0')%a;}
    return ((b[n-1]-'0')%a + ((10%a)*mod(b,a,n-1))%a )%a;
}
