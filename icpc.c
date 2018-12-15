

#include <stdio.h>
#include<stdlib.h>
#include<math.h>
long long int merge(long long int *a,long long int p,long long int q ,long long int r){
 long long int *l;
 long long int *k;
 l=(long long int*)malloc((q-p+1)*sizeof(long long int));
 k=(long long int*)malloc((r-q)*sizeof(long long int));
 long long int i,j;
 for ( i = 0; i < q-p+1; i++)
   l[i] = a[p + i];
 for ( j = 0; j < r-q; j++)
   k[j] = a[q + 1+ j];

 i=0;
 j=0;
 long long int x=p;
long long int count=0;
 while (i < q-p+1 && j < r-q)
 {
   if (l[i] >2* k[j])
   {
 count+=q-p+1-i;
     a[x] = k[j];
     j++;
   }
   else
   { a[x] = l[i];
     i++;

   }
   x++;
 }
 i=0;
j=0;
x=p;
 while (i < q-p+1 && j < r-q)
 {
   if (l[i] <= k[j])
   {
     a[x] = l[i];
     i++;
   }
   else
   {
     a[x] = k[j];
     j++;
   }
   x++;
 }

 while (i < q-p+1)
 {
   a[x] = l[i];
   i++;
   x++;
 }
  while (j < r-q)
 {
   a[x] = k[j];
   j++;
   x++;
 }
return count;
}
long long int sort(long long int *a,long long int p,long long int r){
 long long int q;
long long int l1=0;
long long int l2=0;
long long int count=0;
 if(r>p){
   q =floor((p+r)/2);
    l1=sort(a,p,q);
    l2=sort(a,q+1,r);
    count=merge(a,p,q,r);
}
return l1+l2+count;

}
int main(){
/* long long int num;
 scanf("%d", &num);            // Reading input from STDIN
 printf("Input number is %d.\n", num);    // Writing output to STDOUT*/
 long long int test;
 scanf("%lld",&test);
 for(long long int cases =0;cases<test; cases++){
   long long int no;
   scanf("%lld",&no);
   long long int *a;
   a=(long long int*)malloc(no*sizeof(long long int));
   for(long long int i=0;i<no;i++){
     scanf("%lld",&a[i]);
   }
   long long int ans=sort(a,0,no-1);
   /* for(long long int i=0;i<no;i++){
     printf("%d ",a[i]);
   }*/
printf("%lld\n",ans);
 }

}
