/*
// Sample code to perform I/O:

cin >> name;                            // Reading input from STDIN
cout << "Hi, " << name << ".\n";        // Writing output to STDOUT

// Warning: Printing unwanted or ill-formatted data to output will cause the test cases to fail
*/

// Write your code here
#include<iostream>

typedef struct Cordinates
{
    int a1,a2,a3,dist;
}pair;

typedef struct Queue
{
    int tail=0,head=0,size = 500;
    int A,B,C,a,b,c,n,t1,t2,t3;
    pair** xyz = new pair*[500];
    bool ***array;
    void Enqueue(pair* x)
    {
      std::cout<< head <<' '<<tail<<':'<< x->a1<<' '<< x->a2<<' '<< x->a3<<' ' << '\n';
      // if(head==2)
      xyz[tail++] = x;
      tail = tail%(size);
    }
    pair* Dequeue()
    {
      pair *x = xyz[head];
      head++;
      head = head%size;
      std::cout<< head <<' '<<tail<<':'<< x->a1<<' '<< x->a2<<' '<< x->a3<<' ' << '\n';
      return x;
    }

    bool isvalid(pair * x)
    {
      // if(x->a1 <0 || x->a1>A || x->a2< 0 || x->a2>B || x->a3<0 || x->a3>C)return false;
      if(t1+t2+t3 > (x->a1 + x->a2 + x->a3)) return false;
      if(array[x->a3][x->a2][x->a1]==true ) return false;
      return true;
    }

    int BFS()
    {
        pair * u = new pair;
        u->a1 = a; u->a2 = b; u->a3 = c; u->dist = 0;
        array[u->a3][u->a2][u->a1]=true;
        Enqueue(u);
        while(head!=tail)
        {
           u = Dequeue();
           // std::cout<< head <<' '<<tail<<':'<< u->a1<<' '<< u->a2<<' '<< u->a3<<' ' << '\n';
           // std::cout << head <<' '<<tail<< '\n';

           if(u->a1 == t1 && u->a2 == t2 && u->a3 == t3 ) {std::cout << "finished" << '\n';return u->dist;}

           // 1->2
           if(a<B-b)
           {
             pair *x = new pair;
                        x->a1 = 0; x->a2 = a+b; x->a3 = c;
             if((array[x->a3][x->a2][x->a1]!=true))
                {  Enqueue(x);
                  std::cout << "1->2" << '\n';
                  array[x->a3][x->a2][x->a1]=true; x->dist = u->dist + 1; }
             else delete x;
           }
           else{
             pair *x = new pair;    x->a1 = a+b-B; x->a2 = B; x->a3 = c;
             if((array[x->a3][x->a2][x->a1]!=true)){ Enqueue(x); array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
               // std::cout << "1->2" << '\n';
             else delete x;
           }
           // 1 -> 3
           // std::cout <<"1->3 "<<' '<<tail<< '\n';
           if(a<C-c)
           {
             pair *x = new pair;
                        x->a1 = 0; x->a2 = b; x->a3 = a + c;
             if((array[x->a3][x->a2][x->a1]!=true)){ Enqueue(x);
               // std::cout << "1->3" << '\n';
                array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           else
           {
             pair *x = new pair;
                        x->a1 = a+c-C; x->a2 = b; x->a3 = C;
             if((array[x->a3][x->a2][x->a1]!=true)){ Enqueue(x);
               std::cout << "1->3" << '\n';
                array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           //2 -> 3
           if(b<C-c)
           {

             pair *x = new pair;
                        x->a1 = a; x->a2 = 0; x->a3 = b + c;
             if((array[x->a3][x->a2][x->a1]!=true)){ Enqueue(x);
std::cout << "2->3" << '\n';
                array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           else
           {
             pair *x = new pair;
                        x->a1 = a; x->a2 = b+c-C; x->a3 = C;
             if((array[x->a3][x->a2][x->a1]!=true)){ Enqueue(x);
std::cout << "2->3" << '\n';
                array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           // 2-> 1
           if(b<A-a)
           {
             pair *x = new pair;
                        x->a1 = a+b; x->a2 = 0; x->a3 = c;
             if((array[x->a3][x->a2][x->a1]!=true)){ Enqueue(x);
std::cout << "2->1" << '\n';
                array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           else
           {
             pair *x = new pair;
                        x->a1 = A; x->a2 = a+b-A; x->a3 = c;
             if((array[x->a3][x->a2][x->a1]!=true)){ Enqueue(x);
std::cout << "2->1" << '\n';
               array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           // 3 -> 1
           if(c<A-a)
           {
             pair *x = new pair;
                        x->a1 = a+c; x->a2 = b; x->a3 = 0;
             if((array[x->a3][x->a2][x->a1]!=true)){ Enqueue(x);
std::cout << "3->1" << '\n';
                array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           else
           {
             pair *x = new pair;
                        x->a1 = A; x->a2 = b; x->a3 = c+a-A;
             if((array[x->a3][x->a2][x->a1]!=true)){ Enqueue(x);
std::cout << "3->1" << '\n';
                array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           // 3 -> 2
           if(c<B-b)
           {
             pair *x = new pair;
                        x->a1 = a; x->a2 = b+c; x->a3 = 0;
             if((array[x->a3][x->a2][x->a1]!=true)){ Enqueue(x);
               std::cout << "3->2" << '\n';

               array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           else
           {
             pair *x = new pair;
                        x->a1 = a; x->a2 = B; x->a3 = b+c-B;
             if((array[x->a3][x->a2][x->a1]!=true)){ Enqueue(x);
               std::cout << "3->1" << '\n';
                array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           // 1->ground
           if(a!=0)
           {
             pair *x = new pair;
                        x->a1 = 0; x->a2 = b; x->a3 = c;
             if((array[x->a3][x->a2][x->a1]!=true)){ Enqueue(x); array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           if(b!=0)
           {
             pair *x = new pair;
                        x->a1 = a; x->a2 = 0; x->a3 = c;
             if((array[x->a3][x->a2][x->a1]!=true)){ Enqueue(x); array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           if(c!=0)
           {
             pair *x = new pair;
                        x->a1 = a; x->a2 = b; x->a3 = 0;
             if((array[x->a3][x->a2][x->a1]!=true)){ Enqueue(x); array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1;  }
             else delete x;
           }
           delete u;
        }
    }

}queue;


int main()
{
    int m;
    std::cin>>m;
    while(m--)
    {
        queue Q;
        std::cin>>Q.A>>Q.C>>Q.B;
        std::cin>>Q.a>>Q.c>>Q.b;

        std::cin>>Q.n;
        while(Q.n--)
        {
            std::cin>>Q.t1>>Q.t3>>Q.t2;
            if(Q.t1+Q.t2+Q.t3>Q.a+Q.b+Q.c) std::cout << "IMPOSSIBLE" << '\n';
            else
            {
                bool ***arr= new bool**[Q.C+1];
                Q.array =arr;
                for(int i=0;i<=Q.C;i++)
                {
                  arr[i] = new bool*[Q.B+1];
                  for(int j=0;j<=Q.B;j++)
                  {
                    arr[i][j] = new bool[Q.A+1];
                    for(int k=0;k<=Q.A;k++)
                    {
                         arr[i][j][k]=false;
                         // pair * w = new pair;
                         // w->a1 = k; w->a2  = j; w->a3 = i;
                         // std::cout<<i<<' '<<j<<' '<<k<<' '<< Q.isvalid(w)<< '\n';
                    }
                  }
                }
                std::cout <<Q.BFS()<< '\n';
                // Q.dist = 0;
            }

            // Solve(arr,A,B,C,a,b,c,a,b,c);
        }
    }
}
