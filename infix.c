/*
// Sample code to perform I/O:
#include <stdio.h>

int main(){
	int num;
	scanf("%d", &num);              			// Reading input from STDIN
	printf("Input number is %d.\n", num);       // Writing output to STDOUT
}

// Warning: Printing unwanted or ill-formatted data to output will cause the test cases to fail
*/

// Write your code here
#include<stdio.h>

int is_higher_priority(char a,char b)
{
    if(a==b)
    {
        if(a=='^') return 1;
        return 0;
    }
    if(a=='+' && b=='-' || a=='-' && b=='+') return 0;
    if(a=='%' && b=='/' || a=='/' && b=='%') return 0;
    char arr[] = {'^','%','/','*','+','-'};
    int i=0,j=0;
    for(;i<6;i++) if(a==arr[i]) break;
    for(;j<6;j++) if(b==arr[j]) break;
    return i>j ? 0:1;
}

int main()
{
    int t;
    scanf("%d\n",&t);
    char optr[1000];
    while(t--)
    {
        char inp[1000],out[1000];
        int head=0;
        scanf("%[^\n]\n",inp);
        int i=0,k=0,operators = 0,operands = 0;
        while(inp[i]!='\0')
        {
            if(inp[i]>='0' && inp[i]<='9')
            {
                while(inp[i]>='0' && inp[i]<='9') out[k++] = inp[i++];
                out[k++] =' ';
                i--;
                operands++;
            }
            else if(inp[i]!=' ')
            {
                operators++;
                if(head==0) optr[head++]=inp[i];
                else if(inp[i]=='(') optr[head++] = inp[i];
                else if(inp[i]==')')
                {
                    operators-=2;
                    head--;
                    while(optr[head]!='(')
                    {
                        out[k++] = optr[head--];//printf("%c ",optr[head--]);
                        out[k++] =' ';
                    }
                    optr[head] = inp[i];
                }
                else if(is_higher_priority(inp[i],optr[head-1])) optr[head++] = inp[i];
                else
                {
                    while(head>0 && !is_higher_priority(inp[i],optr[head-1]) )
                    {
                        head--;
                        out[k++] = optr[head];//printf("%c ",optr[head]);
                        out[k++] =' ';
                        optr[head] = inp[i];
                    }
                    head++;
                }
            }
            i++;
        }
        head--;
        while(head>=0){out[k++] = optr[head--]; out[k++] =' ';}//printf("%c ",optr[head--])
        if(operators == (operands - 1) )
        {
            for(int u=0;u<k;u++)
            {
                printf("%c",out[u]);
            }
            printf("\n");
        }
        else printf("INVALID\n");
    }
}
