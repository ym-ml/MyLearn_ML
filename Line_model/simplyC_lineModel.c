#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define M_E  2.71828182845904523536
#define Alpha 0.0005
#define Xrow 2
#define MINI 1e-12

double *multied_w0_X(double a[],int a_size,int b[][3],int x_row ){
   
    double *result=(double*)calloc(x_row,sizeof(double));
    if (result==NULL){
        printf("destributing wrong");
        return NULL;
    }

    for (int i=0;i<x_row;i++){
        for (int j=0;j<a_size;j++){
            result[i]+=a[j]*b[i][j];
        }
    }
    return result;
} 

double get_loss(int y[],int y_size,double p_1[]){
    double loss=0;
    for (int i=0;i<y_size;i++){
        loss+=y[i]*log(p_1[i]+MINI)
                +(1-y[i])*log(1-p_1[i]+MINI);
    }
    return -loss/Xrow;

}

double *multied_XYp(int x[][3],int xsize,int y[],double p_1[]){
    static double delta_w[3]={0};
    for (int i=0;i<3;i++){
        for (int j=0;j<xsize;j++){
            delta_w[i]+=1/(double)xsize*Alpha*x[j][i]*(y[j]-p_1[j]);//!
        }
    }
 
    return delta_w;

}

int main(){
    int X[Xrow][3]={{1,0,0},
                 {1,1,1}};//intercept,x1,x2
    int y[Xrow]={0,1};
    double w0[3]={0.5,0.5,0.5};

    double *mult_w0X;
    double p_1[Xrow]={0.};
    double loss;
    
    double *delta_w;


    mult_w0X=multied_w0_X(w0,3,X,Xrow); //remember to free
    if (mult_w0X==NULL){
        printf("Error,interuptted");
        return 0;
    }
    for (int i=0;i<Xrow;i++){
        p_1[i]=pow(M_E,mult_w0X[i])/(1.+pow(M_E,mult_w0X[i]));        

    }
    free(mult_w0X);
//start train
    for (int loop=0;loop<1000;loop++){
        delta_w=multied_XYp(X,Xrow,y,p_1);
        for (int i=0;i<3;i++){
            w0[i]=w0[i]+delta_w[i];
        }
        
        mult_w0X=multied_w0_X(w0,3,X,Xrow); //remember to free
        if (mult_w0X==NULL){
        printf("Error,interuptted");
        return 0;
        }
        for (int i=0;i<Xrow;i++){
            p_1[i]=pow(M_E,mult_w0X[i])/(1.+pow(M_E,mult_w0X[i]));        
        }
        free(mult_w0X);

        if (loop%100==0){
            loss=get_loss(y,Xrow,p_1);
            printf("loop %d ,loss %f\n",loop,loss);
        }
    }

    for (int i=0;i<3;i++)printf("%f ",w0[i]);


    return 0;


}


