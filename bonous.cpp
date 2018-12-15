/*
// Sample code to perform I/O:

#include <iostream>

using namespace std;

int main() {
	int num;
	cin >> num;										// Reading input from STDIN
	cout << "Input number is " << num << endl;		// Writing output to STDOUT
}

// Warning: Printing unwanted or ill-formatted data to output will cause the test cases to fail
*/

// Write your code here
#include<iostream>

using namespace std;
struct stack
{
    int head = 0;
    int *arr;
    void push(int e)
    {
        arr[head++] = e;
    }
    void pop()
    {
        head--;
    }
    int top()
    {
        return arr[head-1];
    }
};

int main()
{
    int t,m;
    cin>>t;
    while(t--)
    {
        cin>>m;
        int *s= new int(m+2);
        struct stack t,u,lowest;
        t.arr = new int(m+2);
        u.arr = new int(m+2);
        lowest.arr = new int(m+2);

        for(int i=0;i<m;i++)
        {
            cin>>s[i];
            if(i==0 || lowest.head==0) lowest.push(i);
            else if(s[i-1]<=s[i]) lowest.push(i);
            else
            {
                while(lowest.head>0 && s[i]<s[lowest.top()]) lowest.pop();
                lowest.push(i);
            }
        }
        // for(int i=0;i<m;i++) cout<<'a'<<s[i]; cout<<'\n';

        int j = 0;
        for(int i=0;i<lowest.head;i++)
        {
            if(t.head>0 && s[lowest.arr[i]] >= t.top() )
            {
                while(t.head>0 && s[lowest.arr[i]] >= t.top() )
                {
                    u.push(t.top());
                    t.pop();
                }
            }

            for(int i=0;i<m;i++) cout<<'f'<<s[i]; cout<<'\n';
            while(j<=lowest.arr[i]) { t.push(s[j++]);}
            // for(int i=0;i<m;i++) cout<<'s'<<s[i]; cout<<'\n';

            if(t.head>0)
            {
                u.push(t.top());
                t.pop();
            }
        }

        while(t.head>0) {u.push(t.top()); t.pop();}

        for(int i=0;i<m;i++) cout<<'v'<<u.arr[i];
        cout<<'\n';
        // for(int i=0;i<m;i++) cout<<'v'<<s[i];
    }

    return 0;
}
