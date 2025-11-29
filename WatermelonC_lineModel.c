#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M_E  2.71828182845904523536
#define Alpha 0.01
#define Xrow 100
#define Testrow 20
#define MINI 1e-12
#define LOOP 30000 //when reach min mean_loss (LOOP=40000),overfitting!


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
void print_watermelon_data(float x[][3], int y[],int y_test[], int size) {
    if (y==NULL){
        printf("编号\t截距\t密度\t\t含糖量\t\t标签\n");
        printf("----------------------------------------------------\n");
        for (int i = 0; i < size; i++) {
            printf("%d\t%.0f\t%.3f\t\t%.3f\n", 
               i + 1, 
               x[i][0],  // 截距项，总是1
               x[i][1],  // 密度
               x[i][2]  // 含糖量
                );    // 标签(空)
        }

    }else if (y!=NULL &&y_test==NULL){
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
    }else if(y!=NULL &&y_test!=NULL){
        printf("编号\t截距\t密度\t\t含糖量\t\t标签\t\t预测标签\tRight Or Wrong\n");
        printf("------------------------------------------------------------------------\n");
        for (int i = 0; i < size; i++) {
            printf("%d\t%.0f\t%.3f\t\t%.3f\t\t%d\t\t%d", 
               i + 1, 
               x[i][0],  // 截距项，总是1
               x[i][1],  // 密度
               x[i][2],  // 含糖量
               y[i],    // 标签
               y_test[i]);  //preduction
            if (y[i]!=y_test[i]){
                printf("\t\twrong");
            }
            printf("\n");
        }
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

void predict (float x_test[][3],double w[],int prediction[],int testsize){
    double tem,pre;
    for (int i=0;i<testsize;i++){
        tem=0.;
        pre=0.;
        for (int j=0;j<3;j++){
            tem+=x_test[i][j]*w[j];
        }
        pre+=1/(1+pow(M_E,-tem));
        if (pre>0.5)prediction[i]=1;
        else prediction[i]=0;
    }
}

int main(){
    char input;
    int user_prediction[5];
    char s;
    printf("这是一个通过西瓜密度和含糖量预测西瓜是否成熟的线性分类器\n");
    printf("请输入任意字符开始：");
    scanf("%s",&s);

    
//set data
    srand(42);                       
    
    // 创建特征矩阵x和标签向量y
    float X[Xrow][3];  // 100×3矩阵，第一列是截距1
    int y[Xrow];       // 100个标签
    
    float X_test[Testrow][3];
    int y_test[Testrow];

    // 生成数据
    generate_watermelon_data(X, y, Xrow);//train
    generate_watermelon_data(X_test,y_test,Testrow);//test

    // 打印前20个样本
    printf("西瓜数据训练集 （1代表熟瓜，0代表生瓜）:\n");
    print_watermelon_data(X, y,NULL, 10);
    printf("西瓜数据预测集：");
    print_watermelon_data(X_test,NULL,NULL,5);

    printf("你想和计算机比一下吗？（输入P开始比较，输入其余字符则放弃）\n");
    scanf("%s",&input);
    if (input =='p' ||input =='P'){
        printf("观察上面那个数据集，总结规律，给出你对预测集的判断：");
        
        for (int i=0;i<5;i++){
            scanf("%d",&user_prediction[i]);
        }
        printf(" \n我记住你的答案了。但是固然你很厉害，而我有100个数据\n");
    }else{
        printf("haha,放弃是对的,我其实有100个数据\n");
    }

    
    printf("这是训练集的全貌（虽然100个也很少了）：\n");   
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

    printf("请输入任意字符，我要开始了\n");
    scanf("%s",&s);

//begin to train

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

    for (int loop=0;loop<LOOP;loop++){
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

        if (loop%(LOOP/10)==0){
            loss=get_loss(y,Xrow,p_1);
            mean_loss+=loss;
            printf("loop %d ,loss %f,mean loss %f\n",loop,loss,mean_loss/((float)(loop/(LOOP/10))+1));
        }
    }    
    
    printf("\n学得参数:\n");
    for (int i=0;i<3;i++){
        printf("%f ",w0[i]); 
    }
    printf("\n");

//test
    int prediction_y[Testrow];
    double smError=0.;
    // 打印测试集
    printf("西瓜数据测试集(20个) :\n");
    print_watermelon_data(X_test, y_test,NULL, Testrow);
    
    predict(X_test,w0,prediction_y,Testrow);

    printf("西瓜数据测试集预测结果:\n");
    print_watermelon_data(X_test, y_test,prediction_y, Testrow);
    
    //square mean error
    for (int i=0;i<Testrow;i++){
        smError+=pow(y_test[i]-prediction_y[i],2);
    }
    printf("均方误差：%.6f",smError/Testrow);
    // 计算准确率
    int correct = 0;
    for (int i = 0; i < Testrow; i++) {
        if (y_test[i] == prediction_y[i]) {
            correct++;
        }
    }
    printf("\n准确率: %.2f%% (%d/%d)\n", (float)correct/Testrow*100, correct, Testrow);

    int score=0;
    if (input=='P'||input =='p'){
        printf("还记得你的预测吗？");
        for (int i=0;i<5;i++){
            printf("  %d:",user_prediction[i]);
            if (y_test[i]==user_prediction[i]){
                score+=1;
                printf("right");
            }
            else printf("wrong");
        }
        printf("\nyour scores:%d\n",score);
        printf("真厉害呀，你得了%d分\n",score);
        
    }

    printf("不过，虽然我用了100个数据，但是我做了20个预测");

    return 0;


}


