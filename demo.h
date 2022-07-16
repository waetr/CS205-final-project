//
// Created by lenovo on 2022/6/18.
//

#ifndef CPP_PROJECT_DEMO_H
#define CPP_PROJECT_DEMO_H

#include "MyMatrix.h"
#include <cstdlib>

namespace demo {
    dense::DenseMat<complex<double> > complexInvertible() {
        dense::DenseMat<complex<double>> mat(3, 3);
        complex<double> t1(1, 0);
        mat.set(1, 1, t1);
        mat.set(3, 2, t1);
        mat.set(3, 3, t1);
        complex<double> t2(1, 1);
        mat.set(1, 3, t2);
        mat.set(2, 2, t2);
        complex<double> t3(0, 1);
        mat.set(2, 1, t3);
        complex<double> t4(2, -1);
        mat.set(1, 2, t4);
        complex<double> t5(1, 2);
        mat.set(2, 3, t5);
        complex<double> t6(-1, 1);
        mat.set(3, 1, t6);
        return mat;
    }

    dense::DenseMat<double> doubleInvertible() {
        dense::DenseMat<double> mat(3, 3);
        mat.set(1, 1, 1);
        mat.set(1, 2, 2);
        mat.set(1, 3, 3);
        mat.set(2, 1, 0);
        mat.set(2, 2, 1);
        mat.set(2, 3, 4);
        mat.set(3, 1, 5);
        mat.set(3, 2, 6);
        mat.set(3, 3, 0);
        return mat;
    }

    dense::DenseMat<double> doubleEigen() {
        dense::DenseMat<double> mat(3, 3);
        mat.set(1, 1, 3);
        mat.set(1, 2, 2);
        mat.set(1, 3, 4);
        mat.set(2, 1, 2);
        mat.set(2, 2, 0);
        mat.set(2, 3, 2);
        mat.set(3, 1, 4);
        mat.set(3, 2, 2);
        mat.set(3, 3, 3);
        return mat;
    }

    dense::DenseMat<double> doubleEigen2() {
        dense::DenseMat<double> mat(2, 2);
        mat.set(1, 1, -1);
        mat.set(1, 2, 4);
        mat.set(2, 1, 2);
        mat.set(2, 2, 6);
        return mat;
    }

    dense::DenseMat<int> intArr(int n, int m) {
        dense::DenseMat<int> mat(n, m);
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                mat.set(i, j, rand() % 10 - 5);
        return mat;
    }

    dense::DenseMat<int> intSimple(int n, int m) {
        dense::DenseMat<int> mat(n, m);
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                mat.set(i, j, (i - 1) * m + j);
        return mat;
    }

    dense::DenseMat<double> doubleArr(int n, int m) {
        dense::DenseMat<double> mat(n, m);
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                mat.set(i, j, (double) (rand() % 100) / 10 - 5);
        return mat;
    }

    dense::DenseMat<complex<double>> complexArr(int n, int m) {
        dense::DenseMat<complex<double>> mat(n, m);
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++) {
                complex<double> z((double) (rand() % 100) / 10 - 5, (double) (rand() % 100) / 10 - 5);
                mat.set(i, j, z);
            }
        return mat;
    }

    dense::DenseMat<dense::DenseMat<int>> matrixArr(int n, int m) {
        dense::DenseMat<dense::DenseMat<int>> mat(n, m);
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++) {
                dense::DenseMat<int> z(2, 2);
                z.set(1, 1, rand() % 10 - 5);
                z.set(1, 2, rand() % 10 - 5);
                z.set(2, 1, rand() % 10 - 5);
                z.set(2, 2, rand() % 10 - 5);
                mat.set(i, j, z);
            }
        return mat;
    }

    void print_matmat(dense::DenseMat<dense::DenseMat<int>> &x) {
        cout << "matMat=\n";
        for (int i = 1; i <= x.row(); i++) {
            cout << "{";
            for (int j = 1; j <= x.col(); j++) {
                dense::DenseMat<int> c = x.get(i, j);
                string str = "";
                for (int I = 1; I <= c.row(); I++) {
                    str += "(";
                    for (int J = 1; J <= c.col(); J++) {
                        str += to_string(c.get(I, J));
                        if (J != c.col()) str += ",";
                    }
                    str += ")";
                }
                cout << "[" << str << "]";
                if (j != x.col())cout << ",";
            }
            cout << "}\n";
        }
        cout << "\n";
    }

    void start_demo() {
        srand((unsigned) time(nullptr));
        using namespace dense;
        using namespace sparse;

        DenseMat<int> one(2, 2);
        cout << "one=\n" << one << endl;

        cout << "input for dense matrix:\n";
        DenseMat<int> mat;
        mat.input();
        cout << "mat=\n" << mat << "input end\n";

        cout << "input for sparse matrix:\n";
        SparseMat<int> sp;
        sp.input();
        cout << "sp=\n" << sp << "input end\n";

        mat = intArr(3, 4);
        cout << "mat=\n" << mat << endl;

        SparseMat<int> tr(mat);
        cout << "tr=\n" << tr << endl;

        mat = SparseToDense(tr);
        cout << "mat=\n" << mat << endl;

        DenseMat<int> matInt = intArr(3, 3);
        DenseMat<complex<double>> matComplex = complexArr(3, 3);
        DenseMat<DenseMat<int>> matMat = matrixArr(3, 3);

        cout << "matInt=\n" << matInt << endl;
        cout << "matComplex=\n" << matComplex << endl;
        print_matmat(matMat);

        matInt = 2 * (matInt * matInt + matInt);
        matComplex = 2.0 * (matComplex * matComplex + matComplex);
        matMat = 2 * (matMat * matMat + matMat);

        cout << "After calculation:\n";
        cout << "matInt=\n" << matInt << endl;
        cout << "matComplex=\n" << matComplex << endl;
        print_matmat(matMat);

        DenseMat<int> a = intSimple(3, 2);
        DenseMat<int> b = intSimple(2, 4);
        DenseMat<int> c = 3 * intSimple(3, 2);
        DenseMat<int> A = intSimple(5,5), T1 = intSimple(5, 1),T2 = intSimple(5, 1);
        cout << "a=\n" << a << endl;
        cout << "b=\n" << b << endl;
        cout << "c=\n" << b << endl;
        cout << "a+c=\n" << a + c << endl;
        cout << "a-c=\n" << a - c << endl;
        cout << "a*2=\n" << a * 2 << endl;
        cout << "T(a)=\n" << a.trans() << endl;
        cout << "bitwise a*c=\n" << a.element_wise_multi(c) << endl;
        cout << "a*b=\n" << a * b << endl;
        cout<<"A*T1=\n"<<A*T1<<endl;
        cout<<"dot product of T1 and T2="<<T1.trans()*T2<<endl;
        cout<<"cross product of T1 and T2=\n"<<T1*T2.trans()<<endl;
        DenseMat<complex<double>> co = complexArr(4,3);
        cout<<"co=\n"<<co<<endl;
        cout<<"conjugation of co=\n"<<co.conj<double>()<<endl;


        DenseMat<double> a0 = doubleArr(3, 4);
        cout << "a0=\n" << a0 << endl;
        cout << "max of rows of a0=\n" << a0.max('x') << endl;
        cout << "max of columns of a0=\n" << a0.max('y') << endl;
        cout << "max of all of a0=" << a0.max('a') << endl;
        cout << "min of all of a0=" << a0.min('a') << endl;
        cout << "sum of all of a0=" << a0.sum('a') << endl;
        cout << "average of all of a0=" << a0.average('a') << endl;

        DenseMat<double> d0 = doubleInvertible();
        DenseMat<complex<double> > c0 = complexInvertible();
        cout << "double matrix=\n"<<d0<<endl;
        cout << "Inverse of double matrix=\n"<<d0.inverse()<<endl;
        cout << "complex matrix=\n"<<c0<<endl;
        cout << "Inverse of complex matrix=\n"<<c0.inverse()<<endl;
        cout << "trace of double matrix="<<d0.trace()<<endl;
        cout << "trace of complex matrix="<<c0.trace()<<endl;
        cout << "Determinant of double matrix="<<d0.det()<<endl;
        cout << "Determinant of complex matrix="<<c0.det()<<endl;
        double res[10];
        d0.eigenvalues(res);
        cout << "Eigenvalue of double matrix:";
        for(int i = 0; i < d0.col(); i++) cout << res[i] << "   ";
        cout<<endl;
        cout << "Eigenvector of double matrix=\n"<<d0.eigenvectors(res)<<endl;

        //6. reshape and slicing
        DenseMat<int> B = intArr(3,4);
        cout << "B=\n"<<B<<endl;
        DenseMat<int> B0 = B.reshape(2, 6);
        cout << "B after reshape=\n"<<B0<<endl;
        B = B0.reshape(3, 4);
        cout << "B reshape again=\n"<<B<<endl;
        B0 = B.slicing(1,3,2,3);
        cout << "B after slicing=\n"<<B0<<endl;

        //7.convolutional operations

        DenseMat<int> C = intArr(4,4);
        cout<<"C=\n"<<C<<endl;
        cout << "convolutional of itself=\n"<<C.conv(C)<<endl;

        //8. convert to openCV()

//        DenseMat<int> I0 = intArr(3,3);
//        DenseMat<double> I1 = doubleArr(3,3);
//        cv::Mat C0 = parseToOpenCV(I0);
//        cv::Mat C1 = parseToOpenCV(I1);
//        cout << "openCVmatrix(int):" << C0.type() << endl << C0 << endl;
//        cout << "openCVmatrix(double):" << C1.type() << endl << C1 << endl;
//        auto Q0 = parseToMatrix<int>(C0);
//        auto Q1 = parseToMatrix<double>(C1);
//        cout << "Q0=\n"<<Q0<<endl;
//        cout << "Q1=\n"<<Q1<<endl;

        //9.exception

        try{
            DenseMat<int> e(-1,-1);
        }catch (exception &e) {
            cout << e.what() << endl;
        }

    }

}

#endif //CPP_PROJECT_DEMO_H
