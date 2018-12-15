//
#include <stdio.h>

void swap(int * a , int * b)
{
  if(*a<=*b) return;
  int temp;
  temp = *a;
  *a = *b;
  *b = temp;
}
int main()
{
  int test;
  scanf("%d",&test);
  while(test--)
  {
    int n,k,u=0,max=-9999,maxindex=0;
    long long int sum=0;
    scanf("%d %d",&n,&k);
    int arr[n];
    int b[n];
    for(int i=0;i<n;i++)
    {
      scanf("%d",arr + i);
      sum+=arr[i];
      if(arr[i]>k)
       {
         b[u++]=arr[i];
         if(arr[i]>max) { max = arr[i]; maxindex = u-1;}
       }
    }
    if(maxindex!=u-1) swap(b+maxindex,b+u-1);
    long int x = 0;
    for(int i=0;i<u-1;i++)
    {
       x += (b[i] - k);
    }
    sum = sum - x - (x%2);
    printf("%lld\n",sum);
  }

}
