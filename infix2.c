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
    char arr[] = {'~','^','%','/','*','+','-'};
    int i=0,j=0;
    for(;i<7;i++) if(a==arr[i]) break;
    for(;j<7;j++) if(b==arr[j]) break;
    return i>j ? 0:1;
}

int check(char *inp)
{
    int ob = 0;
    // int operands=1;

    for(int i=0;inp[i]!='\0';i++)
    {
        if(inp[i]==' ') continue;
        // if(inp[i] == '~')
        // {
        //     if (operands== 1) return 0;
        //     else continue;
        // }
        if(inp[i]=='(')
        {
            ob++;
            // if(operands==1) continue;
            // else return 0;
        }
        else if(inp[i]==')')
        {
            ob--;
            if(ob<0  ) return 0;//|| operands==1
            else continue;
        }
    }

    //     if(operands==1)
    //     {
    //         if( inp[i]>='0' && inp[i]<='9')
    //         {
    //             while(inp[i]!='\0' && inp[i]>='0' && inp[i]<='9') i++;
    //             operands = 0; i--;
    //         }
    //         else { return 0;}
    //     }
    //     else
    //     {
    //         if(inp[i]>='0' && inp[i]<='9') {return 0;}
    //         else if(inp[i]!='(' && inp!=')')
    //         {
    //             operands=1;
    //         }
    //     }

    // }
    if(ob!= 0) return 0;
    return 1;

}

void replace_unary(char * str)
{
    for(int i=0;str[i]!='\0';i++)
    {
        if(str[i]=='-')
        {
            if(i==0 ) str[i]='~';
            else if(str[i-2] == '+'||str[i-2] == '-'||str[i-2] == '%'||str[i-2] == '/'||str[i-2] == '*'||str[i-2] == '^'|| str[i-2]=='(')
            str[i] = '~';
            // else if(!( (str[i+2]>='0' && str[i+2] <='9')||str[i+2]=='('||str[i+2]=='-') )//str[i+1]!='\0' && str[i+2]!='\0'&&
            // {str[i]='~';}//
            // else if(!(str[i-2]  >='0' && str[i-2] <='9' ||str[i-2]==')')) { str[i]='~';}
            // else if(str[i+2]=='(' && (str[i-2]=='*'||str[i-2]=='^'||str[i-2]=='%'||str[i-2]=='/'||str[i-2]=='+'||str[i-2]=='-')) str[i]='~';
        }
    }
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
        replace_unary(inp);
        // printf("%s\n",inp);
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
                if(head==0) { optr[head++]=inp[i]; if(inp[i]=='~') operators--;}
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
                else if(is_higher_priority(inp[i],optr[head-1])) { optr[head++] = inp[i]; if(inp[i]=='~') operators--;}
                else
                {
                    if(inp[i]=='~') operators--;
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
        // printf("%d %d\n",operators,operands);
        if((operands == (operators +1))&& check(inp))//
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
