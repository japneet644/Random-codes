#include<iostream>

using namespace std;

typedef struct Data
{
  int k;
  struct Data * next = NULL;
}data;
struct stack
{
  int a1,a2;
};
typedef struct Node
{
  int k,d=0,low=999990,parent=-1;
  struct Data * child = NULL;
  struct Data * next = NULL;
  struct Data * back = NULL;
  bool col = false;
}node;

typedef struct Graph
{
  struct Node *V;
  int m,n,time = 0,points =0;
  int * articulations;
  void Insert_edge(node * x, int q)
  {
      if(x->next==NULL) {x->next = new data; x->next->k = q; return;}
      data * ptr = x->next;
      while(ptr->next) ptr = ptr->next;
      ptr->next = new data;
      ptr->next->k = q;
  }
  void Insert_child(node * x, int q)
  {
      if(x->child == NULL) { x->child = new data; x->child->k = q; return;}
      data * ptr = x->child;
      while(ptr->next) ptr = ptr->next;
      ptr->next = new data;
      ptr->next->k = q;
  }
  void Insert_back(node * x, int q)
  {
      if(x->back==NULL) {x->back = new data; x->back->k = q; return;}
      data * ptr = x->back;
      while(ptr->next) ptr= ptr->next;
      ptr->next = new data;
      ptr->next->k = q;
  }
  void Print()
  {
    for(int i=0;i<this->m;i++)
    {
      data * ptr = this->V[i].child;
      std::cout << V[i].k<<' '<<V[i].d<<' '<<V[i].low<< " child ";
      while(ptr){   cout<<ptr->k<<' ';   ptr = ptr->next;}
      cout<<'\n';
    }
  }
  void DFS()
  {
    for(int i=0;i<m;i++)
      if(V[i].col == false)  DFS_VISIT(i);
  }
  void DFS_VISIT(int s)
  {
    time+=1;
    V[s].d = time;
    V[s].col = true;
    data* ptr = V[s].next;
    while(ptr)
    {
      if(V[ptr->k].col == false)
      {
        Insert_child(V+s,ptr->k);
        V[ptr->k].parent = s;
        DFS_VISIT(ptr->k);
      }
      else
      {
          if(ptr->k!= V[s].parent && V[s].d> V[ptr->k].d) Insert_back(V+s,ptr->k);
      }
      ptr = ptr->next;
    }
    // time+=1;
  }
  void articulated_points(int i)
  {
      V[i].col =true;
      if(V[i].parent == -1)
      {
          if(V[i].child == NULL || V[i].child->next == NULL) {V[i].low =V[i].d; return;}
          // V[i].low = min();//up................
          data *ptr = V[i].child; V[i].low = V[i].d;
          while(ptr)
          {
            if(V[i].low >  V[ptr->k].low) V[i].low = V[ptr->k].low;
            ptr = ptr->next;
          }
          articulations[points++] = V[i].k;
          return;
      }
      if(V[i].child == NULL)
      {
        V[i].low = V[i].d;
        if(V[i].back == NULL)  return;
        // V[i].low = minofbackedges();//up................
        data * ptr = V[i].back;
        while(ptr)
        {
          if(V[i].low > V[ptr->k].d) V[i].low = V[ptr->k].d;
          // std::cout << "returned"<<V[i].low<<V[ptr->k].low<<'\n';
          ptr = ptr->next;
        }
        return;
      }
      data* ptr = V[i].child;
      while(ptr)
      {
        if(V[ptr->k].col==false)
        {
          // std::cout << "recursion" << V[ptr->k].k<<'\n';
        articulated_points(V[ptr->k].k);}
        ptr= ptr->next;
      }
      ptr = V[i].child; V[i].low = V[i].d; int max=-1; int min = V[i].d;
      while(ptr)
      {
        if(max <  V[ptr->k].low) max = V[ptr->k].low;
        if(min >  V[ptr->k].low) min = V[ptr->k].low;
        ptr = ptr->next;
      }
      if(max > V[i].d)
      {
        // std::cout << " message " << '\n';
        articulations[points++] = V[i].k;
      }
      // else {
        V[i].low = min;
        ptr = V[i].back;
        while(ptr)
        {
          if(V[i].low >V[ptr->k].low) V[i].low = V[ptr->k].low;
          ptr = ptr->next;
        }
      // }
      // V[i.low] = minofbackedges(V[i].low,);
  }
  void sort()
  {
    int count = 1,temp;
    while (count!=0)
    {
      count = 0;
      for(int i=0;i<points-1;i++)
      {
        if(articulations[i]>articulations[i+1])
        {
           temp = articulations[i];
           articulations[i] = articulations[i+1];
           articulations[i+1] = temp; count = 1;
        }
      }
    }
  }
  void Sort_bridge(struct stack** s,int c)
  {
    int count = 1;
    struct stack *temp;
    while (count!=0)
    {
      count = 0;
      for(int i=0;i<c-1;i++)
      {
        if(s[i]->a1 > s[i+1]->a1 || s[i]->a1 == s[i+1]->a1 && s[i]->a2 > s[i+1]->a2)
        {
           temp = s[i];
           s[i] = s[i+1];
           s[i+1] = temp; count = 1;
        }
      }
    }
  }
  void Bridges()
  {
    int count;
    data * ptr;
    struct stack**s = new struct stack*[n];
    count = 0;
    for(int i=0;i<m;i++)
    {
      ptr = V[i].child;
        while(ptr)
        {
            // if(ptr->k > i &&(V[ptr->k].col ==true || ptr->next!=NULL))
            if(ptr->k > i)
            {
              if( V[i].d < V[ptr->k].low )
              { //std::cout <<V[i].low << V[ptr->k].low  << '\n';
                s[count] = new struct stack; s[count]->a1 = i; s[count++]->a2 = ptr->k;}
            }
            else
            {
              if( V[i].d < V[ptr->k].low )
              { //std::cout <<V[i].low << V[ptr->k].low  << '\n';
                s[count] = new struct stack; s[count]->a2 = i; s[count++]->a1 = ptr->k;}

            }
            ptr = ptr->next;
        }

    }
      Sort_bridge(s,count);
      // std::cout << count << '\n';
      for(int i=0;i<count;i++) std::cout << s[i]->a1 <<' '<<s[i]->a2<< '\n';

  }
}graph;

int main()
{
    int p,q;
    graph G;
    cin>>G.m;
    G.V = new node[G.m];
    G.articulations = new int[G.m];
    for(int i=0;i<G.m;i++) G.V[i].k = i;
    cin>>G.n;
    for(int i=0;i<G.n;i++){ cin>>p>>q;  G.Insert_edge(G.V + p,q);G.Insert_edge(G.V+q,p); }
    G.DFS();
    for(int i=0;i<G.m;i++) G.V[i].col = false;
    for(int i=0;i<G.m;i++) if(G.V[i].col==false) G.articulated_points(i);
    G.sort();
    for(int i=0;i<G.points;i++) std::cout << G.articulations[i]<<' ';
    std::cout << '\n';
    for(int i=0;i<G.m;i++) G.V[i].col = false;
    for(int i=0;i<G.points;i++)  G.V[G.articulations[i]].col = true;
    // G.Print();
    G.Bridges();
    return 0;
}
