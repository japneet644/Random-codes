/*


*/

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
      xyz[tail++] = x;
      tail = tail%(size);
    }
    pair* Dequeue()
    {
      pair *x = xyz[head];
      head++;
      head = head%size;
      return x;
    }
    bool isvalid(pair * x)
    {
      if(t1+t2+t3 > x->a1 + x->a2 + x->a3)  return false;
      if(array[x->a3][x->a2][x->a1]==true ) return false;
      return true;
    }
    pair* BFS()
    {
        pair * u = new pair;
        u->a1 = a; u->a2 = b; u->a3 = c; u->dist = 0;
        array[u->a3][u->a2][u->a1]=true;
        Enqueue(u);
        while(head!=tail)
        {
           u = Dequeue();
           if(u->a1 == t1 && u->a2 == t2 && u->a3 == t3 ) return u;
           a = u->a1; b = u->a2; c = u->a3;
           if(a<B-b)
           {
              pair *x = new pair;
                        x->a1 = 0; x->a2 = a+b; x->a3 = c;
              if(isvalid(x))
              { Enqueue(x); x->parent = u; array[x->a3][x->a2][x->a1]=true; x->dist = u->dist + 1; }
              else delete x;
           }
           else
           {
             pair *x = new pair;
                        x->a1 = a+b-B; x->a2 = B; x->a3 = c;
             if(isvalid(x)){ Enqueue(x);x->parent = u; array[x->a3][x->a2][x->a1]=true; x->dist = u->dist + 1; }
             else delete x;
           }
           if(a<(C-c))
           {
             pair *x = new pair;
                        x->a1 = 0; x->a2 = b; x->a3 = a + c;
             if(isvalid(x)){ Enqueue(x);x->parent = u; array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           else
           {
             pair *x = new pair;
                        x->a1 = a+c-C; x->a2 = b; x->a3 = C;
             if(isvalid(x)){ Enqueue(x);x->parent = u; array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           if(b<C-c)
           {
             pair *x = new pair;
                        x->a1 = a; x->a2 = 0; x->a3 = b + c;
             if(isvalid(x)){ Enqueue(x);x->parent = u; array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           else
           {
             pair *x = new pair;
                        x->a1 = a; x->a2 = b+c-C; x->a3 = C;
             if(isvalid(x)){ Enqueue(x);x->parent = u; array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           if(b<A-a)
           {
             pair *x = new pair;
                        x->a1 = a+b; x->a2 = 0; x->a3 = c;
             if(isvalid(x)){ Enqueue(x);x->parent = u; array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           else
           {
             pair *x = new pair;
                        x->a1 = A; x->a2 = a+b-A; x->a3 = c;
             if(isvalid(x)){ Enqueue(x); x->parent = u; array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           if(c<A-a)
           {
             pair *x = new pair;
                        x->a1 = a+c; x->a2 = b; x->a3 = 0;
             if(isvalid(x)){ Enqueue(x);x->parent = u;  array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           else
           {
             pair *x = new pair;
                        x->a1 = A; x->a2 = b; x->a3 = c+a-A;
             if(isvalid(x)){ Enqueue(x);x->parent = u; array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           if(c<B-b)
           {
             pair *x = new pair;
                        x->a1 = a; x->a2 = b+c; x->a3 = 0;
             if(isvalid(x)){ Enqueue(x);x->parent = u; array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           else
           {
             pair *x = new pair;
                        x->a1 = a; x->a2 = B; x->a3 = b+c-B;
             if(isvalid(x)){ Enqueue(x);x->parent = u; array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           if(a!=0)
           {
             pair *x = new pair;
                        x->a1 = 0; x->a2 = b; x->a3 = c;
             if(isvalid(x)){ Enqueue(x);x->parent = u; array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           if(b!=0)
           {
             pair *x = new pair;
                        x->a1 = a; x->a2 = 0; x->a3 = c;
             if(isvalid(x)){ Enqueue(x);x->parent = u; array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1; }
             else delete x;
           }
           if(c!=0)
           {
             pair *x = new pair;
                        x->a1 = a; x->a2 = b; x->a3 = 0;
             if(isvalid(x)){ Enqueue(x);x->parent = u; array[x->a3][x->a2][x->a1]=true;    x->dist = u->dist + 1;  }
             else delete x;
           }
        }
        return NULL;
    }

}queue;


int main()
{
    int m,d,f,g;
    std::cin>>m;
    while(m--)
    {
        queue Q;
        std::cin>>Q.A>>Q.C>>Q.B;
        std::cin>>Q.a>>Q.c>>Q.b;
        d = Q.a; f = Q.b; g = Q.c;
        std::cin>>Q.n;

        while(Q.n--)
        {
            std::cin>>Q.t1>>Q.t3>>Q.t2;
            if(Q.t1+Q.t2+Q.t3 > Q.a+Q.b+Q.c) std::cout << "IMPOSSIBLE" << '\n';
            else
            {
                bool ***arr= new bool**[Q.C+1];
                for(int i=0;i<=Q.C;i++)
                {
                arr[i] = new bool*[Q.B+1];
                  for(int j=0; j<=Q.B; j++)
                  {
                    arr[i][j] = new bool[Q.A+1];
                    for(int k=0; k<=Q.A; k++)   arr[i][j][k]=false;
                  }
                }
                Q.array = arr;
                pair * u = Q.BFS();
                if(u) std::cout <<u->dist+1<<'\n';

                while(u)
                {
                  std::cout<<u->a1<<' '<< u->a2<<' '<< u->a3<<' ' << '\n';
                  u = u->parent;}
                Q.a = d; Q.b = f; Q.c = g;
                Q.head = Q.tail;
            }
        }
    }
}
