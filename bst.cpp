#include<iostream>

using namespace as std;
struct node
{
   int key;
   struct node* right = NULL,*left = NULL;
}node;

node * top;

void insert(node * n, int a)
{
   node * cr = top;
   if(cr==NULL){top = n;  top->key = a; return; }
   while(1)
   {
     if(cr->left)
     {
       if(a <= cr->left->key)    cr = cr->left;
       else
       {
         if(cr->right){cr = cr->right;}
         else
         {
           cr->right = n;
           cr->right->key = a;
           return;
         }
       }
     }
     else
     {
       cr->left = n;
       cr->left->key = a;
       return;
     }
   }
}

node* search(int a)
{
  if(head==NULL)return NULL;
  node* cr = head;
  while(1)
  {
    if(cr->key)==a)
        return cr;
    else if(a < cr->key)
    {
        if(cr->left)
             cr = cr->left;
        else
             return NULL;
    }
    else
    {
      if(cr->right) cr = cr->right;
      else return NULL;
    }
  }
  return NULL;
}

void del(int a)
{
  if(head==NULL) cout<<"not present empty tree :("<<endl;
  node* cr = head,*prev =NULL;
  while(1)
  {
    if(cr->key)==a)
    {
      node* x;

      delete cr;
      cout<<"deleted:)"<<endl;
    }
    else if(a < cr->key)
    {
        if(cr->left)
        {
             prev =cr;
             cr = cr->left;
        }
        else
             cout<<"not present"<<endl;
    }
    else
    {
      if(cr->right) {prev =cr; cr = cr->right;}
      else cout<<"not present :("<<endl;
    }
  }
  cout<<"not present :("<<endl;

}
int main()
{
  node * top = NULL;

  return 0;
}
