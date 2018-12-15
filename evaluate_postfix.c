#include<stdio.h>

int perform_operation(int a,int b, char c)
{
    int x;
    switch(c)
    {
        case '+': return a+b;
        case '-': return a-b;
        case '*': return a*b;
        case '/': x=(float)a/b; return (int)x;
        case '%': return a%b;
        case '^': {x = 1; for(int i=0;i<b;i++) x *=a;  return x;}
    }
}
int main()
{
    int t;
    scanf("%d\n",&t);
    while(t--)
    {
        char str[1000];
        int stack[100];
        scanf("%[^\n]\n",str);
        int i=0,head =0;
        while(str[i]!='\0')
        {
            if(str[i]>='0'&&str[i]<='9')
            {
                int x = str[i++]-'0';
                while(str[i]>='0' && str[i]<='9')
                {
                    x = 10*x + (int)(str[i++]-'0');
                }
                i--;
                stack[head++]=x;
            }
            else if(str[i]!=' ')
            {
                if(head>1){
                    if((str[i]=='/'|| str[i]=='%') && stack[head-1] ==0) {printf("INVALID\n"); break;}
                    else
                    {
                        stack[head-2] = perform_operation(stack[head-2],stack[head-1],str[i]); head--;
                    }

                }
                //printf("%d\n",stack[head-1]);}
                else {printf("INVALID\n"); break;}
            }
            i++;
        }
        if(head!=1) printf("INVALID\n");
        else printf("%d\n",stack[head-1]);
    }
    return 0;
}
