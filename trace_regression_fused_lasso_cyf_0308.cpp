#include <RcppArmadillo.h>
#include <iostream>
#include <armadillo>
#include <math.h>


// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace std;
using namespace arma;

//Cost function
double costfunC(vec Y, cube X, mat B){
    double out = 0;
    int n = Y.size();
    double temp;
    for (int i = 0; i < n; i++){
        temp = Y(i) - trace(X.slice(i).t() * B);
        out = out + pow(temp,2);
    }
    out = out / n;
    return out;
}

//Nuclear norm penalty
double nuclear_penC(mat B, double lambda1){
    mat U;
    mat V;
    vec d;
    svd(U,d,V,B);
    double out = lambda1 * accu(d);
    return out;
}

//Weight function
mat weightC(mat B, int k, double zeta){
    int p = B.n_rows;
    mat weight(p,p,fill::zeros);
    mat I_mat(p,p,fill::zeros);
    mat exp_part(p,p,fill::zeros);

    double temp;
    for (int i = 0; i < p - 1; i++){
        for (int j = i + 1; j < p; j++){
            temp = arma::norm(B.row(i)-B.row(j),2);
            temp = -zeta * pow(temp,2);
            temp = exp(temp);
            exp_part(i,j) = temp;
        }
    }

    //1 and 0 are set reversely in the following matrix, since we need to find if i is in j's kth neighbor OR vise versa.
    //If we reverse the logic operator 1 and 0 and dot multiply I_temp and t(I_temp), what we want will be 0.
    mat I_temp(p,p,fill::ones);
    vec I_vec(p,fill::zeros);
    for (int i = 0; i < p; i++){
        for (int j = 0; j < p; j++){
            I_vec(j) = arma::norm(B.row(i)-B.row(j),2);
        }
        uvec indice_sort = sort_index(I_vec);
        uvec indice_k = indice_sort.rows(1,k); //Extract the minimal distance (i.e., i itself)
        for (int l = 0; l < k; l++){
            I_temp(i,indice_k(l)) = 0;
        }
    }
    I_temp = I_temp % I_temp.t();
    I_mat = -I_temp + 1;

    //Set the lower triangle to be 0 (i.e., make I an upper triangle matrix)
    for (int j = 0; j < p; j++){
        for (int i = j; i < p; i++){
            I_mat(i,j) = 0;
        }
    }

    weight = I_mat % exp_part;
    return weight;
}

//Fused lasso penalty for 1 dimension
double fused_lasso_pen_1dC(mat B, double lambda2, int k, double zeta){
    double out = 0;
    int p = B.n_rows;
    mat weight = weightC(B,k,zeta);
    for (int i = 0; i < p - 1; i++) {
        for (int j = i + 1; j < p; j++) {
            out = out + weight(i,j) * arma::norm(B.row(i)-B.row(j),2);
        }
    }
    out = out * lambda2;
    return out;
}

//Fused lasso penalty for 2 dimensions (B and t(B))
double fused_lasso_pen_2dC(mat B, double lambda2, double lambda3, int k_row, int k_col, double zeta){
    double out = 0;
    out = fused_lasso_pen_1dC(B,lambda2,k_row,zeta) + fused_lasso_pen_1dC(B.t(),lambda3,k_col,zeta);
    return out;
}

//Objective function
double objfunC(vec Y, cube X, mat B, double lambda1, double lambda2, double lambda3, int k_row, int k_col, double zeta){
    double out = 0;
    out = costfunC(Y,X,B) + nuclear_penC(B,lambda1) + fused_lasso_pen_2dC(B,lambda2,lambda3,k_row,k_col,zeta);
    return out;
}



//Input k and output i, so that we can calculate j
int k2ijC(int k, int p){
    int flag = 0;
    for (int i = p - 1; i >= 0; i--){
        if (k - i >= 0){
            k = k - i;
            flag++;
        }
        else{
            break;
        }
    }
    return flag;
}

//Proximal for update U and V
vec prox_L2(vec v, double sigma){
    int n = v.size();
    int i;
    vec prox(n,fill::zeros);
    double sum=0.0;
    for (i=0; i<n; i++){
        sum = sum + pow(v(i),2.0);
    }
    double l2norm=sqrt(sum);
    if (sum==0.0){
        for (i=0; i<n; i++)
            prox(i)=v(i);
    }
    else{
        for(i=0; i<n; i++)
            prox(i)=fmax(0.0, 1.0-(sigma/l2norm))*v(i);
    }
    return prox;
}



//Gradient of f(B)
mat gradfC(vec Y, cube X, mat B, mat U, mat V, mat G, mat H, mat E1, mat E2, double rho){
    int p = B.n_rows;
    int q = B.n_cols;
    mat out;
    mat temp(p,q,fill::zeros);
    int n = Y.size();
    for (int i = 0; i < n; i++) {
        temp = temp + X.slice(i) * (Y(i) - trace(X.slice(i).t() * B));
    }
    temp = -2 * temp / n;
    out = temp - rho * E1.t() * (U - E1 * B + G) - rho * (V - E2 * B.t() + H).t() * E2;
    return out;
}

//Function to solve P_tk(B)
mat solvepC(vec Y, cube X, mat B, mat U, mat V, mat G, mat H, mat E1, mat E2, double lambda1, double rho, double tk){
    mat temp = B - gradfC(Y,X,B,U,V,G,H,E1,E2,rho) / tk;
    mat UU;
    mat VV;
    vec d;
    svd(UU,d,VV,temp);
    vec soft_d = d - (lambda1/tk);
    soft_d.elem(find(soft_d < 0)).fill(0);
    int p = d.size();
    mat out = UU.cols(0, p - 1) * diagmat(soft_d) * VV.cols(0, p - 1).t();
    return out;
}

//Value of Q(B,B_old)
double QvalueC(vec Y, cube X, mat B, mat B_old, mat U, mat V, mat G, mat H, mat E1, mat E2,
               double lambda1, double lambda2, double lambda3, int k_row, int k_col, double zeta, double rho, double tk){
    double out = 0;
    out = costfunC(Y,X,B_old) + nuclear_penC(B,lambda1) + fused_lasso_pen_2dC(B_old,lambda2,lambda3,k_row,k_col,zeta);
    out = out + trace((B-B_old).t() * gradfC(Y,X,B_old,U,V,G,H,E1,E2,rho));
    out = out + tk * pow(arma::norm(B-B_old,"fro"),2) / 2;
    return out;
}

//Update B
mat updateBC(vec Y, cube X, mat Z, mat U, mat V, mat G, mat H, mat E1, mat E2, double lambda1, double rho, double tk){
    mat out = solvepC(Y,X,Z,U,V,G,H,E1,E2,lambda1,rho,tk);
    return out;
}

//Update alpha
double update_alphaC(double alpha_old){
    double alpha;
    alpha = (1 + sqrt(1 + 4 * pow(alpha_old,2))) / 2;
    return alpha;
}

//Update Z
mat updateZC(mat B, mat B_old, double alpha, double alpha_old){
    mat Z;
    Z = B + (alpha_old - 1) * (B - B_old) / alpha;
    return Z;
}


//Could be used for updating both U and V
mat updateUVC(mat B, mat G, mat E, mat weight, double lambda, double rho){
    int p = B.n_rows;
    int q = B.n_cols;
    int nK = E.n_rows;
    int i, j;
    mat EB = E*B;
    mat UV(nK,q,fill::zeros);
    vec a(q,fill::zeros);
    vec aa(q,fill::zeros);
    for (int k = 0; k < nK; k++){
        a = (EB.row(k) - G.row(k)).t();
        i = k2ijC(k,p);
        j = k + i + 1 - (2 * p - i - 1) * i / 2;
        aa = prox_L2(a,weight(i,j)*lambda/rho);
        UV.row(k) = aa.t();
    }
    return UV;
}

//Could be used for updating both G and H
mat updateGHC(mat G_old, mat B, mat U, mat E, double rho){
    int q=B.n_cols, nK = E.n_rows;
    arma::mat EB = E*B;
    arma::mat G(nK,q,fill::zeros);
    for (int i=0; i<nK; i++){
        G.row(i) = G_old.row(i) - rho * (EB.row(i) - U.row(i));
    }
    return G;
}


// [[Rcpp::export("TRFL_C")]]
List TRFL_C(vec Y, cube X, mat B, mat U, mat V, mat G, mat H, mat E1, mat E2,
          int max_iter, double tol, double lambda1, double lambda2, double lambda3,
          double rho,double tk, double gamma, int k_row, int k_col, double zeta){
    int iter;
    double t_bar;
    double alpha = 1;
    double alpha_old;
    double f = 1e9;
    double f_old;
    mat Z = B;
    mat weight1, weight2;

    for (iter = 0; iter <= max_iter; iter++){
        mat G_old = G;
        mat H_old = H;
        mat B_old = B;
        alpha_old = alpha;
        f_old = f;


        t_bar = tk;

        while (objfunC(Y,X,solvepC(Y,X,Z,U,V,G,H,E1,E2,lambda1,rho,t_bar),lambda1,lambda2,lambda3,k_row,k_col,zeta)
               > QvalueC(Y,X,solvepC(Y,X,Z,U,V,G,H,E1,E2,lambda1,rho,t_bar),Z,U,V,G,H,E1,E2,lambda1,lambda2,lambda3,k_row,k_col,zeta,rho,t_bar)){
            t_bar = gamma * t_bar;
        }

        tk = t_bar;
        B = solvepC(Y,X,Z,U,V,G,H,E1,E2,lambda1,rho,tk);
        alpha = update_alphaC(alpha_old);
        Z = updateZC(B,B_old,alpha,alpha_old);
        weight1 = weightC(B,k_row,zeta);
        weight2 = weightC(B.t(),k_col,zeta);
        U = updateUVC(B,G,E1,weight1,lambda2,rho);
        V = updateUVC(B.t(),H,E2,weight2,lambda3,rho);
        G = updateGHC(G_old,B,U,E1,rho);
        H = updateGHC(H_old,B.t(),V,E2,rho);

        f = objfunC(Y,X,B,lambda1,lambda2,lambda3,k_row,k_col,zeta);

        if (f > f_old){
            break;
        }
        else{
            if (fabs(f - f_old) / f_old < tol){
                break;
            }
        }
    }

    List output = List::create(_["B"]=wrap(B),
                               _["objective"]=wrap(f),
                               _["iter"]=wrap(iter));
    return output;
}