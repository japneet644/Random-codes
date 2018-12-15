/*
// Sample code to perform I/O:

cin >> name;                            // Reading input from STDIN
cout << "Hi, " << name << ".\n";        // Writing output to STDOUT

// Warning: Printing unwanted or ill-formatted data to output will cause the test cases to fail
*/

// Write your code here
#include<iostream>
using namespace std;

bool strcompare(char *a,char * b)
{
    int x=0;
    while(a[x]!='\0'||b[x]!='\0')
    {
        if(a[x]!=b[x]) return false;
        x++;
    }
    if(a[x]==b[x]) return true;
    return false;
}

typedef struct Interval
{
    int i1;
    int i2;
}interval;

typedef struct Node
{
  struct Interval i;
  int max;
  struct Node * parent;
  struct Node * right;
  struct Node * left;
    Node()
    {
        struct Node * parent=NULL;
        struct Node * right=NULL;
        struct Node * left=NULL;
    }

}node;

int maximum(int a, int b , int c)
{
    return (a>b)?(a>c?a:c):(b>c?b:c);
}

typedef struct Tree
{
    node * root;
    Tree(){this->root = NULL; }
    void Transplant(node * a,node *b)
    {
        if(a->parent == NULL) {this->root=b; return; }
        if(a->parent->left==a)a->parent->left = b;
        else a->parent->right = b;
        if(b) b->parent = a->parent;
    }
    void Insert(interval x)
    {
        // cout<<"Insert"<<x.i1<<' '<<x.i2<<'\n';
        node * ptr = (this->root);
        if(this->root==NULL) {ptr = new node; ptr->i=x; ptr->max = x.i2; this->root = ptr; return;}

        node * prev = (this->root);
        ptr->max = (x.i2>ptr->max)? x.i2 : ptr->max;
        if(x.i1 < ptr->i.i1) ptr=ptr->left;
        else ptr=ptr->right;
        while(ptr)
        {
            prev = ptr;
            ptr->max = (x.i2>ptr->max) ? x.i2:ptr->max;
            if(x.i1 < ptr->i.i1)  ptr=ptr->left;
            else ptr=ptr->right;
        }
        ptr = new node;
        ptr->parent= prev;
        ptr->i=x;
        ptr->max = x.i2;
        if(x.i1 < prev->i.i1) prev->left = ptr;
        else prev->right = ptr;
    }

    void Delete(interval x)
    {
        node * ptr = (this->root);
        node * u;
        if(ptr==NULL) return;

        while(ptr && !(ptr->i.i1==x.i1))
        {
            if(x.i1 < ptr->i.i1)  ptr=ptr->left;
            else ptr=ptr->right;
        }
        if(ptr==NULL) return;
        if(ptr->left==NULL){u = ptr->right; Transplant(ptr,ptr->right); }
        else if(ptr->right==NULL) { u= ptr->left; Transplant(ptr,ptr->left); }
        else
        {
            node * y = ptr->right;
            while(y->left!=NULL) y=y->left;
            if(y->parent!=ptr)
            {
                u = y->right;
                Transplant(y,y->right);
                y->right = ptr->right;
                y->right->parent = y;
            }
            Transplant(ptr,y);
            y->left = ptr->left;
            y->left->parent = y;
        }
        while(u)
        {
            u->max = maximum(u->parent->i.i2,u->parent->left->i.i2,u->parent->right->i.i2);
            u = u->parent;
        }
    }

    interval min()
    {
        node * ptr = (this->root);
        while(ptr->left) ptr=ptr->left;
        return ptr->i;
    }

    interval max()
    {
        node * ptr = (this->root);
        int target = ptr->max;
        while(ptr->i.i2 != target)
        {
            if(ptr->left && ptr->left->max==target) ptr = ptr->left;
            else ptr = ptr->right;
        }
        return ptr->i;
    }

    interval LoSucc()
    {

    }

    bool IsOverlap(interval x)
    {
        node * ptr = (this->root);
        if(ptr && ptr->max<x.i1) return false;
        while(1)
        {
            if(ptr->i.i2>x.i1 && ptr->i.i1<x.i2) return true;

            if(ptr->left && ptr->left->max>= x.i1) ptr = ptr->left;
            else if(ptr->right) ptr = ptr->right;
            else return false;
        }
    }
    void Inorder(node * n)
    {
        if (n == NULL)
            return;
        Inorder(n->left);
        cout << n->max << " ";
        Inorder(n->right);
    }

}tree;

int main(){
    int m,n,i1,i2;
    char inp[10];
    cin>>m;
    while(m--)
    {
        cin>>n;
        tree T;
        while(n--)
        {
            cin>>inp;
            // T.Inorder(T.root); cout<<'\n';
            if(strcompare(inp,"+")) { interval x;   cin>>x.i1>>x.i2; T.Insert(x);  }
            else if(strcompare(inp,"-"))    { interval x; cin>>x.i1>>x.i2; T.Delete(x);}
            else if(strcompare(inp,"min"))  { interval x; x = T.min();      cout<<'['<<x.i1<<' '<<x.i2<<']'<<'\n';}
            else if(strcompare(inp,"max"))  { interval x; x = T.max();      cout<<'['<<x.i1<<' '<<x.i2<<']'<<'\n';}
            // else if(strcompare(inp,"lsucc")){ interval x; cin>>x.i1>>x.i2; x = T.LoSucc(x) cout<<'['<<x.i1<<' '<<x.i2<<']'<<'\n';}
            else if(strcompare(inp,"hsucc")){ interval x; cin>>x.i1>>x.i2;}// x = T.HiSucc(x) cout<<'['<<x.i1<<' '<<x.i2<<']'<<'\n';}
            else if(strcompare(inp,"overlap")) { interval x; cin>>x.i1>>x.i2;  cout<<(T.IsOverlap(x)?'1':'0')<<'\n'; }
        }
    }

    return 0;
}
