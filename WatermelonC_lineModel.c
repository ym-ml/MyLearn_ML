#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define M_E  2.71828182845904523536
#define Alpha 0.001
#define Xrow 100
#define MINI 1e-12


// 生成指定范围内的随机浮点数
float random_float(float min, float max) {
    return min + ((float)rand() / RAND_MAX) * (max - min);
}

// 生成西瓜数据集
void generate_watermelon_data(float x[][3], int y[], int size) {
    for (int i = 0; i < size; i++) {
        // 生成密度和含糖量
        float density = random_float(0.5f, 1.0f);
        float sugar = random_float(0.0f, 0.3f);
        
        // 设置x数组：第一列为截距1，第二列为密度，第三列为含糖量
        x[i][0] = 1.0f;      // 截距项
        x[i][1] = density;   // 密度特征
        x[i][2] = sugar;     // 含糖量特征
        
        // 基于特征设置标签（简单的规则）
        // 密度高且含糖量高的更可能是熟瓜
        if (density > 0.75f && sugar > 0.15f) {
            y[i] = 1;  // 熟瓜
        } else if (density < 0.65f && sugar < 0.08f) {
            y[i] = 0;  // 生瓜
        } else {
            // 中间区域随机分配，增加一些噪声
            y[i] = rand() % 2;
        }
    }
}

// 打印数据集
void print_watermelon_data(float x[][3], int y[], int size) {
    printf("编号\t截距\t密度\t\t含糖量\t\t标签\n");
    printf("----------------------------------------------------\n");
    for (int i = 0; i < size; i++) {
        printf("%d\t%.0f\t%.3f\t\t%.3f\t\t%d\n", 
               i + 1, 
               x[i][0],  // 截距项，总是1
               x[i][1],  // 密度
               x[i][2],  // 含糖量
               y[i]);    // 标签
    }
}

double *multied_w0_X(double a[],int a_size,float b[][3],int x_row ){
   
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

double *multied_XYp(float x[][3],int xsize,int y[],double p_1[]){
    static double delta_w[3]={0};
    for (int i=0;i<3;i++){
        for (int j=0;j<xsize;j++){
            delta_w[i]+=1/(double)xsize*Alpha*x[j][i]*(y[j]-p_1[j]);//!
        }
    }
 
    return delta_w;

}

int main(){

    srand(42);
    
    // 创建特征矩阵x和标签向量y
    float X[Xrow][3];  // 100×3矩阵，第一列是截距1
    int y[Xrow];       // 100个标签
    
    // 生成数据
    generate_watermelon_data(X, y, Xrow);
    
    // 打印前20个样本
    printf("西瓜数据集 (前20个样本):\n");
    print_watermelon_data(X, y, 20);
    
    // 统计熟瓜和生瓜的数量
    int ripe_count = 0, unripe_count = 0;
    for (int i = 0; i < Xrow; i++) {
        if (y[i] == 1) {
            ripe_count++;
        } else {
            unripe_count++;
        }
    }
    
    printf("\n数据集统计:\n");
    printf("总样本数: %d\n", Xrow);
    printf("熟瓜数量: %d (%.1f%%)\n", ripe_count, (float)ripe_count/Xrow*100);
    printf("生瓜数量: %d (%.1f%%)\n", unripe_count, (float)unripe_count/Xrow*100);
    
    // 显示x矩阵和y向量的维度信息
    printf("\n数据维度:\n");
    printf("x: %d*%d 矩阵\n", Xrow, 3);
    printf("y: %d 维向量\n\n", Xrow);


    

    double w0[3]={0.5,0.5,0.5};

    double *mult_w0X;
    double p_1[Xrow]={0.};
    double loss,mean_loss=0.;
    
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
    for (int loop=0;loop<10000;loop++){
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

        if (loop%1000==0){
            loss=get_loss(y,Xrow,p_1);
            mean_loss+=loss;
            printf("loop %d ,loss %f,mean loss %f\n",loop,loss,mean_loss/((float)(loop/1000)+1));
        }
    }    

    for (int i=0;i<3;i++)printf("%f ",w0[i]);

    //test , use 20 datas 
    //square mean error

    return 0;


}


