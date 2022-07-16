/*****************************************************************************
*  cs205 project library                                                     *
*  Copyright (C) 2022 Asuka darkash@126.com.                                 *
*                                                                            *
*  This program is free software; you can redistribute it and/or modify      *
*  it under the terms of the GNU General Public License version 3 as         *
*  published by the Free Software Foundation.                                *
*                                                                            *
*  You should have received a copy of the GNU General Public License         *
*  along with OST. If not, see <http://www.gnu.org/licenses/>.               *
*                                                                            *
*  Unless required by applicable law or agreed to in writing, software       *
*  distributed under the License is distributed on an "AS IS" BASIS,         *
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
*  See the License for the specific language governing permissions and       *
*  limitations under the License.                                            *
*                                                                            *
*  @file     MyMatrix.h                                                      *
*  @brief    A header for the class matrix                                   *
*  Details.                                                                  *
*                                                                            *
*  @author   HYQ,XJY,ZJY                                                     *
*  @email    darkash@126.com                                                 *
*  @version  1.0.0.1                                                         *
*  @date     2022-6-18                                                       *
*  @license  GNU General Public License (GPL)                                *
*                                                                            *
*----------------------------------------------------------------------------*
*  Remark         : Description                                              *
*----------------------------------------------------------------------------*
*  Change History :                                                          *
*  <Date>     | <Version> | <Author>       | <Description>                   *
*----------------------------------------------------------------------------*
*  2022/06/18 | 1.0.0.1   | Asuka          | Create file                     *
*----------------------------------------------------------------------------*
*                                                                            *
*****************************************************************************/
#ifndef CPP_PROJECT_MYMATRIX_H
#define CPP_PROJECT_MYMATRIX_H

#include <vector>

#define MAX_ROW 100
#define MAX_COL 100
#define MAX_ROW_SPARSE 100000
#define MAX_COL_SPARSE 100000
using namespace std;

template<class T>
/*!
 * @brief A triple for sparse matrices
 */
struct triple {
    int x, y;
    T v;

    triple(int x, int y, T v) : x(x), y(y), v(v) {}
};

/*!
 * @brief The base class of a matrix.
 */
class Mat {
protected:
    int Row, Col;
public:
    Mat();

    Mat(int row, int col);

    virtual ~Mat();

    int row() const;

    int col() const;
};

Mat::Mat() {
    Row = Col = 1;
}

/*!
 * @brief Set the matrix with row and col.
 * @param[in] row : row number of the matrix
 * @param[in] col : col number of the matrix
 * @exception length_error : row or col too large
 * @exception out_of_range : row or col be not positive
 */
Mat::Mat(int row, int col) {
    if (row > MAX_ROW || col > MAX_COL) throw length_error("Row or column is too large!");
    if (row <= 0 || col <= 0) throw out_of_range("Row or column must be positive!");
    Row = row, Col = col;
}

Mat::~Mat() = default;

int Mat::row() const {
    return Row;
}

int Mat::col() const {
    return Col;
}

/*!
 * @brief A namespace storing DenseMat \n
 */
namespace dense {
    template<class T>
    class DenseMat : public Mat {
    private:
        T *data;

        DenseMat<T> add(const DenseMat<T> &p);

        DenseMat<T> sub(const DenseMat<T> &p);

        DenseMat<T> scalar_multi(double num);

        DenseMat<T> scalar_div(double num);

        T determinant(const DenseMat<T> &p);

        DenseMat<double> QRMa();

    public:
        DenseMat();

        DenseMat(int row, int col);

        DenseMat(int row, int col, T num);

        DenseMat(DenseMat<T> &p);

        virtual ~DenseMat();

        T get(int i, int j) const;

        void set(int i, int j, T v);

        bool operator==(const DenseMat<T> &p);

        bool operator!=(const DenseMat<T> &p);

        DenseMat<T> &operator=(const DenseMat<T> &p);

        DenseMat<T> operator+(const DenseMat<T> &p);

        DenseMat<T> operator-(const DenseMat<T> &p);

        DenseMat<T> operator-();

        friend DenseMat<T> operator*(double num, DenseMat<T> &p) {
            return p * num;
        }

        friend DenseMat<T> operator*(int num, DenseMat<T> p) {
            return p * num;
        }

        /*!
         * @brief Output overloading friend function.
         */
        friend ostream &operator<<(ostream &os, const DenseMat<T> &c) {
            for (int i = 1; i <= c.row(); i++) {
                os << "(";
                for (int j = 1; j <= c.col(); j++) {
                    os << setw(10);
                    os << c.get(i, j);
                    if (j != c.col()) os << ",";
                }
                os << ")\n";
            }
            return os;
        }

        DenseMat<T> operator*(double num);

        DenseMat<T> operator/(double num);

        DenseMat<T> operator*(int num);

        DenseMat<T> operator/(int num);

        DenseMat<T> element_wise_multi(const DenseMat<T> &p);

        DenseMat<T> trans();

        template<class P>
        DenseMat<complex<P>> conj();

        DenseMat<T> operator*(const DenseMat<T> &p);

        DenseMat<T> max(char c = 'a');

        DenseMat<T> min(char c = 'a');

        DenseMat<T> sum(char c = 'a');

        DenseMat<T> average(char c = 'a');

        DenseMat<T> reshape(int row_new, int col_new);

        DenseMat<T> slicing(int x1, int x2, int y1, int y2);

        DenseMat<T> inverse();

        T trace();

        T det();

        T cofactor(int i, int j);

        void eigenvalues(double *res);

        DenseMat<T> conv(DenseMat<T> core);

        DenseMat<double> eigenvectors(const double *eigenValue);

        void input();
    };

    /*!
     * @brief Init the DenseMat with 0 for a default size.
     */
    template<class T>
    DenseMat<T>::DenseMat():Mat() {
        data = new T[1]();
    }

    /*!
     * @brief Init the DenseMat with 0 for a given row * col size.
     * @param[in] row : the row number of DenseMat
     * @param[in] col : the col number of DenseMat
     */
    template<class T>
    DenseMat<T>::DenseMat(int row, int col):Mat(row, col) {
        data = new T[this->row() * this->col() + 1]();
    }

    /*!
     * @brief Copy constructor for DenseMat.
     * @param[in] p : DenseMat to be copied
     */
    template<class T>
    DenseMat<T>::DenseMat(DenseMat<T> &p):DenseMat(p.row(), p.col()) {
        for (int i = 1; i <= row(); i++)
            for (int j = 1; j <= col(); j++)
                set(i, j, p.get(i, j));
    }

    /*!
     * @brief Init the DenseMat with a specific num for a given row * col size.
     * @param[in] row : the row number of DenseMat
     * @param[in] col : the col number of DenseMat
     * @param[in] num : the num to fill into the matrix
     */
    template<class T>
    DenseMat<T>::DenseMat(int row, int col, T num):DenseMat(row, col) {
        for (int i = 1; i <= row; i++)
            for (int j = 1; j <= col; j++)
                set(i, j, num);
    }

    /*!
     * @brief A destructor for DenseMat.
     */
    template<class T>
    DenseMat<T>::~DenseMat() {
        delete[] data;
    }

    /*!
     * @brief Get the value of specific index in matrix.
     * @param[in] i : the row of index
     * @param[in] j : the col of index
     * @note Notice that the index of matrix starts from 1.
     * @return a value of matrix in (i,j)
     * @exception out_of_range : row or col be too small or too large
     */
    template<class T>
    T DenseMat<T>::get(int i, int j) const {
        if (i <= 0 || j <= 0 || i > row() || j > col()) throw out_of_range("Row or column be out of range!");
        return *(data + (i - 1) * col() + j - 1);
    }

    /*!
     * @brief Set the value of specific index in matrix.
     * @param[in] i : the row of index
     * @param[in] j : the col of index
     * @param[in] v : the value to set
     * @note Notice that the index of matrix starts from 1.
     * @exception out_of_range : row or col be too small or too large
     */
    template<class T>
    void DenseMat<T>::set(int i, int j, T v) {
        if (i <= 0 || j <= 0 || i > row() || j > col()) throw out_of_range("Row or column be out of range!");
        *(data + (i - 1) * col() + j - 1) = v;
    }

    /*!
     * @brief An equivalence operator overloading function.
     */
    template<class T>
    bool DenseMat<T>::operator==(const DenseMat<T> &p) {
        if (row() != p.row() || col() != p.col()) return false;
        for (int i = 1; i <= row(); i++)
            for (int j = 1; j <= col(); j++)
                if (!(get(i, j) == p.get(i, j))) return false;
        return true;
    }

    template<class T>
    bool DenseMat<T>::operator!=(const DenseMat<T> &p) {
        return !(*this == p);
    }

    /*!
     * @brief An assignment operator overloading function.
     */
    template<class T>
    DenseMat<T> &DenseMat<T>::operator=(const DenseMat<T> &p) {
        if (this == &p) return *this;

        delete[] data;
        this->Col = p.col();
        this->Row = p.row();
        data = new T[this->row() * this->col() + 1]();
        for (int i = 1; i <= row(); i++)
            for (int j = 1; j <= col(); j++)
                set(i, j, p.get(i, j));
        return *this;
    }

    /*!
     * @brief Support arithmetic addition for matrix.
     * @param[in] p : addend
     * @return a matrix with addition result
     * @exception out_of_range : sizes of matrix are not same
     */
    template<class T>
    DenseMat<T> DenseMat<T>::add(const DenseMat<T> &p) {
        if (p.col() != col() || p.row() != row()) throw out_of_range("In this::Row or column must be same!");
        DenseMat<T> mat(p.row(), p.col());
        for (int i = 1; i <= row(); ++i) {
            for (int j = 1; j <= col(); ++j) {
                mat.set(i, j, get(i, j) + p.get(i, j));
            }
        }
        return mat;
    }

    template<class T>
    DenseMat<T> DenseMat<T>::operator+(const DenseMat<T> &p) {
        return this->add(p);
    }

    /*!
     * @brief Support arithmetic subtraction for matrix.
     * @param[in] p : subtrahend
     * @return a matrix with subtraction result
     * @exception out_of_range : sizes of matrix are not same
     */
    template<class T>
    DenseMat<T> DenseMat<T>::sub(const DenseMat<T> &p) {

        if (p.col() != col() || p.row() != row()) throw out_of_range("Row or column must be same!");

        DenseMat<T> mat(p.row(), p.col());

        for (int i = 1; i <= row(); ++i) {
            for (int j = 1; j <= col(); ++j) {
                mat.set(i, j, get(i, j) - p.get(i, j));
            }
        }
        return mat;

    }

    /*!
     * @brief A subtraction operator overloading function.
     */
    template<class T>
    DenseMat<T> DenseMat<T>::operator-(const DenseMat<T> &p) {
        return this->sub(p);
    }

    template<class T>
    DenseMat<T> DenseMat<T>::operator-() {
        DenseMat<T> res = *this;
        for (int i = 1; i <= row(); ++i) {
            for (int j = 1; j <= col(); ++j) {
                res.set(i, j, -res.get(i, j));
            }
        }
        return res;
    }

    /*!
     * @brief Support arithmetic scalar multiplication for matrix.
     * @param[in] num : scalar to be multiplied
     * @return a matrix with scalar multiplication result
     */
    template<class T>
    DenseMat<T> DenseMat<T>::scalar_multi(double num) {
        DenseMat<T> mat(row(), col());

        for (int i = 1; i <= row(); ++i) {
            for (int j = 1; j <= col(); ++j) {
                T value = this->get(i, j) * num;
                mat.set(i, j, value);
            }
        }
        return mat;
    }

    template<class T>
    DenseMat<T> DenseMat<T>::operator*(double num) {
        return scalar_multi(num);
    }

    template<class T>
    DenseMat<T> DenseMat<T>::operator*(int num) {
        return scalar_multi((double) num);
    }

    /*!
     * @brief Support arithmetic scalar division for matrix.
     * @param[in] num : scalar to be divided
     * @return a matrix with scalar division result
     * @excepton domain_error : divide the matrix by 0
     */
    template<class T>
    DenseMat<T> DenseMat<T>::scalar_div(double num) {

        if (num == 0) throw domain_error("0 cannot exist as a divisor!");

        DenseMat<T> mat(row(), col());
        for (int i = 1; i <= row(); ++i) {
            for (int j = 1; j <= col(); ++j) {
                T value = this->get(i, j) / num;
                mat.set(i, j, value);
            }
        }
        return mat;
    }

    /*!
     * @brief A division operator overloading function.
     */
    template<class T>
    DenseMat<T> DenseMat<T>::operator/(double num) {
        return scalar_div(num);
    }

    template<class T>
    DenseMat<T> DenseMat<T>::operator/(int num) {
        return scalar_div((double) num);
    }

    /*!
     * @brief Support transposition for matrix.
     * @return a matrix with transposition result
     */
    template<class T>
    DenseMat<T> DenseMat<T>::trans() {

        DenseMat<T> mat(col(), row());

        for (int i = 1; i <= row(); ++i) {
            for (int j = 1; j <= col(); ++j) {
                mat.set(j, i, get(i, j));
            }
        }
        return mat;
    }

    /*!
     * @brief Support conjugation for matrix.
     * @return a matrix with conjugation result
     * @note The result can be a complex.
     */
    template<class T>
    template<class P>
    DenseMat<complex<P> > DenseMat<T>::conj() {
        DenseMat<complex<P> > mat(row(), col());
        for (int i = 1; i <= row(); ++i) {
            for (int j = 1; j <= col(); ++j) {
                P imagNum = get(i, j).imag();
                P realNum = get(i, j).real();
                mat.set(i, j, complex<P>(realNum, -imagNum));
            }
        }
        return mat;
    }

    /*!
     * @brief Support element-wise multiplication for matrix.
     * @param[in] p : the matrix to be multiplied by element-wise
     * @return a matrix with element-wise multiplication result
     * @exception out_of_range : sizes of matrix are not same
     */
    template<class T>
    DenseMat<T> DenseMat<T>::element_wise_multi(const DenseMat<T> &p) {

        if (row() != p.row() || col() != p.col()) throw out_of_range("Row or column must be same!");
        DenseMat<T> mat(row(), col());

        for (int i = 1; i <= row(); ++i) {
            for (int j = 1; j <= col(); ++j) {
                T val = get(i, j) * p.get(i, j);
                mat.set(i, j, val);
            }
        }

        return mat;
    }


    /*!
     * @brief A multiplication operator overloading function.
     * @note It can be used in all multiplication operation, for dot product and cross product, etc.
     * @exception domain_error : row of right is not equal to col of left
     */
    template<class T>
    DenseMat<T> DenseMat<T>::operator*(const DenseMat<T> &p) {
        if (this->col() != p.row()) throw domain_error("Row of right is not equal to column of left");
        int n = this->row(), m = this->col(), q = p.col();
        DenseMat<T> result(n, q);
        auto *zero = new DenseMat<T>();
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= q; j++)
                for (int k = 1; k <= m; k++) {
                    if (result.get(i, j) == zero->get(1, 1))
                        result.set(i, j, this->get(i, k) * p.get(k, j));
                    else
                        result.set(i, j, result.get(i, j) + this->get(i, k) * p.get(k, j));
                }
        delete zero;
        return result;
    }

    /*!
     * @brief Get a maximum result according to the input char.
     * @param[in] c : control of the maximum type:
     *                  'x' : maximum of each row
     *                  'y' : maximum of each col
     *                  'a' : maximum of all element
     * @return a matrix with maximum result of given type
     * @exception invalid_argument : char c input is invalid
     */
    template<class T>
    DenseMat<T> DenseMat<T>::max(char c) {
        int n = this->row(), m = this->col();
        DenseMat<T> resultx(n, 1);
        DenseMat<T> resulty(1, m);
        DenseMat<T> result(1, 1);
        switch (c) {
            case 'x':

                for (int i = 1; i <= n; i++) {
                    resultx.set(i, 1, this->get(i, 1));
                    for (int j = 2; j <= m; j++)
                        if (resultx.get(i, 1) < this->get(i, j))
                            resultx.set(i, 1, this->get(i, j));
                }
                break;
            case 'y':
                for (int i = 1; i <= m; i++) {
                    resulty.set(1, i, this->get(1, i));
                    for (int j = 2; j <= n; j++)
                        if (resulty.get(1, i) < this->get(j, i))
                            resulty.set(1, i, this->get(j, i));
                }
                break;
            case 'a':
                for (int i = 1; i <= n; i++) {
                    for (int j = 1; j <= m; j++)
                        if ((i == 1 && j == 1) || result.get(1, 1) < this->get(i, j))
                            result.set(1, 1, this->get(i, j));
                }
                break;
            default:
                throw invalid_argument("the character must be {'x','y','a'}");
        }
        return (c == 'a') ? result : ((c == 'x') ? resultx : resulty);
    }

    /*!
     * @brief Get a minimum result according to the input char.
     * @param[in] c : control of the minimum type:
     *                  'x' : minimum of each row
     *                  'y' : minimum of each col
     *                  'a' : minimum of all element
     * @return a matrix with minimum result of given type
     * @exception invalid_argument : char c input is invalid
     */
    template<class T>
    DenseMat<T> DenseMat<T>::min(char c) {
        int n = this->row(), m = this->col();
        DenseMat<T> resultx(n, 1);
        DenseMat<T> resulty(1, m);
        DenseMat<T> result(1, 1);
        switch (c) {
            case 'x':

                for (int i = 1; i <= n; i++) {
                    resultx.set(i, 1, this->get(i, 1));
                    for (int j = 2; j <= m; j++)
                        if (resultx.get(i, 1) > this->get(i, j))
                            resultx.set(i, 1, this->get(i, j));
                }
                break;
            case 'y':
                for (int i = 1; i <= m; i++) {
                    resulty.set(1, i, this->get(1, i));
                    for (int j = 2; j <= n; j++)
                        if (resulty.get(1, i) > this->get(j, i))
                            resulty.set(1, i, this->get(j, i));
                }
                break;
            case 'a':
                for (int i = 1; i <= n; i++) {
                    for (int j = 1; j <= m; j++)
                        if ((i == 1 && j == 1) || result.get(1, 1) > this->get(i, j))
                            result.set(1, 1, this->get(i, j));
                }
                break;
            default:
                throw invalid_argument("the character must be {'x','y','a'}");
        }
        return (c == 'a') ? result : ((c == 'x') ? resultx : resulty);
    }

    /*!
     * @brief Get a sum result according to the input char.
     * @param[in] c : control of the sum type:
     *                  'x' : sum of each row
     *                  'y' : sum of each col
     *                  'a' : sum of all element
     * @return a matrix with sum result of given type
     * @exception invalid_argument : char c input is invalid
     */
    template<class T>
    DenseMat<T> DenseMat<T>::sum(char c) {
        int n = this->row(), m = this->col();
        DenseMat<T> resultx(n, 1);
        DenseMat<T> resulty(1, m);
        DenseMat<T> result(1, 1);
        switch (c) {
            case 'x':

                for (int i = 1; i <= n; i++) {
                    resultx.set(i, 1, this->get(i, 1));
                    for (int j = 2; j <= m; j++)
                        resultx.set(i, 1, resultx.get(i, 1) + this->get(i, j));
                }
                break;
            case 'y':
                for (int i = 1; i <= m; i++) {
                    resulty.set(1, i, this->get(1, i));
                    for (int j = 2; j <= n; j++)
                        resulty.set(1, i, resulty.get(1, i) + this->get(j, i));
                }
                break;
            case 'a':
                for (int i = 1; i <= n; i++) {
                    for (int j = 1; j <= m; j++)
                        result.set(1, 1, result.get(1, 1) + this->get(i, j));
                }
                break;
            default:
                throw invalid_argument("the character must be {'x','y','a'}");
        }
        return (c == 'a') ? result : ((c == 'x') ? resultx : resulty);
    }

    /*!
     * @brief Get an average result according to the input char.
     * @param[in] c : control of the average type:
     *                  'x' : average of each row
     *                  'y' : average of each col
     *                  'a' : average of all element
     * @return a matrix with average result of given type
     * @exception invalid_argument : char c input is invalid
     */
    template<class T>
    DenseMat<T> DenseMat<T>::average(char c) {
        if (c != 'x' && c != 'y' && c != 'a') throw invalid_argument("the character must be {'x','y','a'}");
        DenseMat<T> result = this->sum(c);
        result = result / (double) ((this->col() * this->row()) / (result.col() * result.row()));
        return result;
    }

    /*!
     * @brief Support reshape operation with a new row and col for the matrix.
     * @param[in] row_new : new row of the matrix
     * @param[in] col_new : new col of the matrix
     * @return a matrix with new row and col size
     * @exception length_error : count of elements of new and old mismatch
     * @exception out_of_range : new row or col is too small or too large
     */
    template<class T>
    DenseMat<T> DenseMat<T>::reshape(int row_new, int col_new) {
        if (row_new > MAX_ROW || col_new > MAX_COL) throw length_error("Row or column is too large!");
        if (row_new <= 0 || col_new <= 0) throw out_of_range("Row or column must be positive!");
        if (row_new * col_new != row() * col()) throw length_error("Count of elements mismatch!");
        DenseMat<T> result = *this;
        result.Row = row_new;
        result.Col = col_new;
        return result;
    }

    /*!
     * @brief Support slice operation by given index(left-down and right-up) for the matrix.
     * @param[in] x1 : row of left-down index
     * @param[in] x2 : row of right-up index
     * @param[in] y1 : col of left-down index
     * @param[in] y2 : col of right-up index
     * @return a sliced piece of matrix
     * @exception out_of_range : the given index is too small or too large
     *                           left-down index is larger than right-up index
     */
    template<class T>
    DenseMat<T> DenseMat<T>::slicing(int x1, int x2, int y1, int y2) {
        if (x1 <= 0 || x2 > row() || x1 > x2) throw out_of_range("Out of x-axis range!");
        if (y1 <= 0 || y2 > col() || y1 > y2) throw out_of_range("Out of y-axis range!");
        DenseMat<T> result(x2 - x1 + 1, y2 - y1 + 1);
        for (int i = x1; i <= x2; i++)
            for (int j = y1; j <= y2; j++)
                result.set(i - x1 + 1, j - y1 + 1, this->get(i, j));
        return result;
    }

    /*!
     * @brief Support the traces calculation operation of the matrix.
     * @return a numeric result of the traces calculation
     * @exception out_of_range : the row and col of the matrix are not the same
     */
    template<class T>
    T DenseMat<T>::trace() {
        if (row() != col()) throw out_of_range("Row and column must be same!");
        T trace = get(1, 1);
        for (int i = 2; i <= col(); i++) {
            trace = trace + get(i, i);
        }
        return trace;
    }

    /*!
     * @brief Support determinant computing operation for the matrix.
     * @param[in] p : the matrix to be computed
     * @return a numeric result of determinant of the matrix
     * @exception out_of_range : the row and col of matrix are not the same
     */
    template<class T>
    T DenseMat<T>::determinant(const DenseMat<T> &p) {

        if (p.col() != p.row()) throw out_of_range("Row and column must be same!");

        if (p.col() == 1) return p.get(1, 1);

        DenseMat<T> bb(p.row() - 1, p.col() - 1);

        T *zero = new T();
        T sum = *zero;
        T flag;

        for (int i = 1; i <= p.col(); ++i) {
            int sign = ((i & 1) == 1) ? 1 : -1;
            flag = p.get(1, i);
            for (int x = 2; x <= p.row(); ++x) {
                for (int y = 1; y < i; ++y) {
                    bb.set(x - 1, y, p.get(x, y));
                }
            }
            for (int x = 2; x <= p.row(); ++x) {
                for (int y = i + 1; y <= p.col(); ++y) {
                    bb.set(x - 1, y - 1, p.get(x, y));
                }
            }

            if (sum == *zero) sum = (sign == 1 ? flag : -flag) * determinant(bb);
            else sum = sum + (sign == 1 ? flag : -flag) * determinant(bb);
        }

        delete zero;
        return sum;
    }

    /*!
     * @brief Return the det result of this matrix.
     */
    template<class T>
    T DenseMat<T>::det() {
        return determinant(*this);
    }

    /*!
     * @brief Support the cofactor computing operation with specific index for the matrix.
     * @param[in] i : the row of the cofactor
     * @param[in] j : the col of the cofactor
     * @return a numeric result of the cofactor of the matrix
     * @exception out_of_range : row and cow of matrix are not the same
     *                           row or col is too small or too large
     */
    template<class T>
    T DenseMat<T>::cofactor(int i, int j) {
        if (col() != row()) throw out_of_range("row and col must be same!");
        if (i <= 0 || j <= 0 || i > row() || j > col()) throw out_of_range("Row or column be out of range!");
        if (col() == 1) return 0;
        DenseMat<T> mat(row() - 1, col() - 1);

        for (int x = 1; x < i; ++x) {
            for (int y = 1; y < j; ++y) {
                mat.set(x, y, get(x, y));
            }
        }
        for (int x = i + 1; x <= row(); ++x) {
            for (int y = 1; y < j; ++y) {
                mat.set(x - 1, y, get(x, y));
            }
        }
        for (int x = 1; x < i; ++x) {
            for (int y = j + 1; y <= col(); ++y) {
                mat.set(x, y - 1, get(x, y));
            }
        }
        for (int x = i + 1; x <= row(); ++x) {
            for (int y = j + 1; y <= col(); ++y) {
                mat.set(x - 1, y - 1, get(x, y));
            }
        }
        return determinant(mat);

    }

    /*!
     * @brief Support the inverse operation for the matrix.
     * @return a matrix with an inverse result
     * @exception out_of_range : row and col of the matrix are not the same
     *                           the matrix is irreversible
     */
    template<class T>
    DenseMat<T> DenseMat<T>::inverse() {
        if (row() != col()) throw out_of_range("Row and column must be same!");
        T det0 = det();
        if (det0 == (T) 0) throw out_of_range("Matrix is irreversible!");
        DenseMat<T> Mt = trans();
        DenseMat<T> coMet(row(), col());
        for (int i = 1; i <= row(); ++i) {
            for (int j = 1; j <= col(); ++j) {
                coMet.set(i, j, Mt.cofactor(i, j));
            }
        }
        DenseMat<T> signMat(coMet.row(), coMet.col());

        for (int i = 1; i <= coMet.row(); ++i) {
            for (int j = 1; j <= coMet.col(); ++j) {
                T sign = ((i + j) & 1) == 1 ? (T) -1 : (T) 1;
                signMat.set(i, j, sign);
            }
        }
        coMet = coMet.element_wise_multi(signMat);
        coMet = coMet.scalar_multi((double) (1 / det0));
        return coMet;
    }

    /*!
     * @brief Support the inverse operation for the complex matrix.
     * @return a complex matrix with an inverse result
     * @exception out_of_range : row and col of the matrix are not the same
     *                           the real part of matrix is irreversible
     *                           the image part of matrix is irreversible
     */
    template<>
    DenseMat<complex<double> > DenseMat<complex<double>>::inverse() {
        if (row() != col()) throw out_of_range("Row and column must be same!");
        DenseMat<double> A(row(), col());
        DenseMat<double> B(row(), col());
        for (int i = 1; i <= row(); ++i) {
            for (int j = 1; j <= col(); ++j) {
                A.set(i, j, this->get(i, j).real());
                B.set(i, j, this->get(i, j).imag());
            }
        }
        if (A.det() == 0) throw out_of_range("The real part is irreversible!");
        if (B.det() == 0) throw out_of_range("The image part is irreversible!");
        DenseMat<double> Real = (A + B * A.inverse() * B).inverse();
        DenseMat<double> Imag = A.inverse() * B * Real;
        DenseMat<complex<double> > ans(row(), col());
        for (int i = 1; i <= row(); ++i) {
            for (int j = 1; j <= col(); ++j) {
                complex<double> z0(Real.get(i, j), -Imag.get(i, j));
                ans.set(i, j, z0);
            }
        }
        return ans;
    }

    /*!
     * @brief  Support QR factorization operation for the matrix.
     * @return the Q matrix result after QR factorization
     * @exception out_of_range : the row and col of the matrix are not the same
     *                           the matrix can not be QR factorized
     */
    template<>
    DenseMat<double> DenseMat<double>::QRMa() {
        if (this->row() != this->col()) throw out_of_range("Matrix must be square!");
        int n = this->row();
        DenseMat<double> Q(this->row(), this->row());
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= n; j++)
                if (i == j) Q.set(i, j, 1.0);
                else Q.set(i, j, 0.0);

        int nn = n - 1;
        double u, alpha, w, t;
        for (int k = 0; k <= nn - 1; k++)//在大循环k：0~m当中，进行H矩阵的求解，左乘Q，以及左乘A
        {
            u = 0.0;
            for (int i = k; i <= n - 1; i++) {
                w = fabs(this->get(i + 1, k + 1));
                if (w > u) u = w;
            }
            alpha = 0.0;
            for (int i = k; i <= n - 1; i++) {
                t = this->get(i + 1, k + 1) / u;
                alpha = alpha + t * t;
            }
            if (this->get(k + 1, k + 1) > 0.0) u = -u;
            alpha = u * sqrt(alpha);
            if (fabs(alpha) + 1.0 == 1.0) throw out_of_range("QR factorization failed!");

            u = sqrt(2.0 * alpha * (alpha - this->get(k + 1, k + 1)));
            if ((u + 1.0) != 1.0) {
                this->set(k + 1, k + 1, (this->get(k + 1, k + 1) - alpha) / u);
                for (int i = k + 1; i <= n - 1; i++)
                    this->set(i + 1, k + 1, this->get(i + 1, k + 1) / u);

                //以上就是H矩阵的求得，实际上程序并没有设置任何数据结构来存储H矩
                //阵，而是直接将u向量的元素赋值给原A矩阵的原列向量相应的位置，这样做
                //这样做是为了计算左乘矩阵Q和A
                for (int j = 0; j <= n - 1; j++) {
                    t = 0.0;
                    for (int jj = k; jj <= n - 1; jj++)
                        t = t + this->get(jj + 1, k + 1) * Q.get(jj + 1, j + 1);
                    for (int i = k; i <= n - 1; i++)
                        Q.set(i + 1, j + 1, Q.get(i + 1, j + 1) - 2.0 * t * this->get(i + 1, k + 1));
                }
                //左乘矩阵Q，循环结束后得到一个矩阵，再将这个矩阵转置一下就得到QR分解中的Q矩阵
                //也就是正交矩阵

                for (int j = k + 1; j <= n - 1; j++) {
                    t = 0.0;
                    for (int jj = k; jj <= n - 1; jj++)
                        t = t + this->get(jj + 1, k + 1) * this->get(jj + 1, j + 1);
                    for (int i = k; i <= n - 1; i++)
                        this->set(i + 1, j + 1, this->get(i + 1, j + 1) - 2.0 * t * this->get(i + 1, k + 1));
                }
                //H矩阵左乘A矩阵，循环完成之后，其上三角部分的数据就是上三角矩阵R
                this->set(k + 1, k + 1, alpha);
                for (int i = k + 1; i <= n - 1; i++) this->set(i + 1, k + 1, 0.0);
            }
        }
        for (int i = 0; i <= n - 2; i++)
            for (int j = i + 1; j <= n - 1; j++) {
                t = Q.get(i + 1, j + 1);//Q[i][j];
                Q.set(i + 1, j + 1, Q.get(j + 1, i + 1));
                Q.set(j + 1, i + 1, t);
            }
        return Q;
    }

    /*!
     * @brief Support eigenvalues computing operation for the matrix.
     * @param[out] res : store the eigenvalues computed
     * @note The elements of matrix can only be double type.
     * @exception out_of_range : the row and col of the matrix are not the same
     */
    template<>
    void DenseMat<double>::eigenvalues(double *res) {
        if (this->row() != this->col()) throw out_of_range("Matrix must be square!");
        int n = this->row();
        if (n == 1) {
            res[0] = this->get(1, 1);
            return;
        }
        DenseMat<double> A1, A2, Q;
        A1 = *this;
        for(int t = 1; t <= 100; t++){
            Q = A1.QRMa();
            A2 = A1 * Q;
            A1 = A2;
        }
        for (int i = 0; i < n; i++) res[i] = A1.get(i + 1, i + 1);
    }

    /*!
     * @brief Support convolutional operations of two matrices.
     * @param[in] core : the convolutional core
     * @return a matrix with convolutional result
     * @exception out_of_range : the row or col of the core is larger than the matrix
     *                           the row and col of the core are not the same
     */
    template<class T>
    DenseMat<T> DenseMat<T>::conv(DenseMat<T> core) {
        if (col() < core.col() || row() < core.row()) throw out_of_range("the core is too large!");
        if (col() != row()) throw out_of_range("the core must be a square!");


        DenseMat mat(row() + 2, col() + 2);

        for (int i = 2; i <= row() + 1; ++i) {
            for (int j = 2; j <= col() + 1; ++j) {
                mat.set(i, j, get(i - 1, j - 1));
            }
        }

        for (int i = 1; i <= col() + 2; ++i) {
            mat.set(1, i, (T) 0);
            mat.set(row() + 2, i, (T) 0);
        }

        for (int i = 1; i <= row() + 2; ++i) {
            mat.set(i, 1, (T) 0);
            mat.set(i, col() + 2, (T) 0);
        }

        for (int i = 1; i <= (core.row() + 1) / 2; ++i) {
            for (int j = 1; j <= core.col(); ++j) {
                T temp = core.get(i, j);
                core.set(i, j, core.get(core.row() - i + 1, core.col() - j + 1));
                core.set(core.row() - i + 1, core.col() - j + 1, temp);
            }
        }

        DenseMat<T> ans(mat.row() - core.row() + 1, mat.col() - core.col() + 1);

        for (int i = 1; i <= mat.row() - core.row() + 1; ++i) {
            for (int j = 1; j <= mat.col() - core.col() + 1; ++j) {
                T sum = (T) 0;
                for (int x = 0; x < core.row(); ++x) {
                    for (int y = 0; y < core.col(); ++y) {
                        sum = sum + mat.get(x + i, y + j) * core.get(x + 1, y + 1);
                    }
                }
                ans.set(i, j, sum);
            }
        }

        return ans;
    }

    /*!
     * @brief Support eigenvectors computing operation of the matrix.
     * @param[in] eigenValue : the eigenvalue to be computed for the eigenvector
     * @return a matrix with computed eigenvector by the given eigenvalue
     */
    template<>
    DenseMat<double> DenseMat<double>::eigenvectors(const double *eigenValue) {
        unsigned i, j, q;
        int count;
        int m;
        const unsigned NUM = this->col();
        double eValue, sum, midSum, mid;
        DenseMat<double> temp(this->row(), this->col()), eigenVector(this->row(), this->col());
        for (count = 0; count < NUM; ++count) {
            //计算特征值为eValue，求解特征向量时的系数矩阵
            eValue = eigenValue[count];
            temp = *this;
            for (i = 0; i < temp.col(); ++i) {
                temp.set(i + 1, i + 1, temp.get(i + 1, i + 1) - eValue);
            }

            //将temp化为阶梯型矩阵
            for (i = 0; i < temp.row() - 1; ++i) {
                mid = temp.get(i + 1, i + 1);
                for (j = i; j < temp.col(); ++j) {
                    temp.set(i + 1, j + 1, temp.get(i + 1, j + 1) / mid);
                }

                for (j = i + 1; j < temp.row(); ++j) {
                    mid = temp.get(j + 1, i + 1);
                    for (q = i; q < temp.col(); ++q) {
                        temp.set(j + 1, q + 1, temp.get(j + 1, q + 1) - mid * temp.get(i + 1, q + 1));
                    }
                }
            }

            midSum = 1.0;
            eigenVector.set(eigenVector.row(), count + 1, 1.0);
            for (m = temp.row() - 2; m >= 0; --m) {
                sum = 0;
                for (j = m + 1; j < temp.col(); ++j) {
                    sum += temp.get(m + 1, j + 1) * eigenVector.get(j + 1, count + 1);
                }
                sum = -sum / temp.get(m + 1, m + 1);
                midSum += sum * sum;
                eigenVector.set(m + 1, count + 1, sum);
            }

            midSum = sqrt(midSum);
            for (i = 0; i < eigenVector.row(); ++i) {
                eigenVector.set(i + 1, count + 1, eigenVector.get(i + 1, count + 1) / midSum);
            }
        }
        return eigenVector;
    }

    /*!
     * @brief Support the input of a matrix.
     */
    template<class T>
    void DenseMat<T>::input() {
        int x, y;
        string s1, s2, s;
        int flag = 1;
        while (true) {
            cout << "Please input the rows and columns.\n";
            cin >> s1 >> s2;
            if (s1 != to_string(stoi(s1)) || s2 != to_string(stoi(s2))) {
                cout << "The input is not integers!\n Do you want to continue? [y/n]\n";
                cin >> s;
                if (s == "y") continue;
                else {
                    flag = 0;
                    break;
                }
            }
            x = stoi(s1), y = stoi(s2);
            if (x <= 0 || x > MAX_ROW || y < 0 || y > MAX_COL) {
                cout << "rows or columns are out of range!\n Do you want to continue? [y/n]\n";
                cin >> s;
                if (s == "y") continue;
                else {
                    flag = 0;
                    break;
                }
            }
            break;
        }
        if (!flag) return;
        DenseMat<T> res(x, y);
        T v;
        cout << "Please input n*m values in order.\n";
        for (int i = 1; i <= x; i++)
            for (int j = 1; j <= y; j++) {
                cin >> v;
                res.set(i, j, v);
            }
        *this = res;
    }
}

/*!
 * @brief A namespace storing SparseMat \n
 */
namespace sparse {
    using namespace dense;

    template<class T>
    class SparseMat : public Mat {
    public:
        std::vector<triple<T> > data_t;

        SparseMat();

        SparseMat(int row, int col);

        SparseMat(SparseMat<T> &p);

        SparseMat(DenseMat<T> &p);

        virtual ~SparseMat();

        T get(int i, int j);

        void set(int i, int j, T v);

        operator DenseMat<T>() const;

        SparseMat<T> &operator=(const SparseMat<T> &p);

        void input();

        friend ostream &operator<<(ostream &os, SparseMat<T> &c) {
            for (int i = 1; i <= c.row(); i++) {
                os << "(";
                for (int j = 1; j <= c.col(); j++) {
                    os << setw(10);
                    os << c.get(i, j);
                    if (j != c.col()) os << ",";
                }
                os << ")\n";
            }
            return os;
        }
    };

    template<class T>
    SparseMat<T>::SparseMat():Mat() {}

    template<class T>
    SparseMat<T>::SparseMat(int row, int col) {
        if (row > MAX_ROW_SPARSE || col > MAX_COL_SPARSE) throw length_error("Row or column is too large!");
        if (row <= 0 || col <= 0) throw out_of_range("Row or column must be positive!");
        Row = row;
        Col = col;
    }

    /*!
     * @brief The constructor using type SparseMat<T>.
     * @tparam T
     * @param p The sparse matrix
     */
    template<class T>
    SparseMat<T>::SparseMat(SparseMat<T> &p):SparseMat(p.row(), p.col()) {
        for (triple<T> e : p.data_t) data_t.push_back(e);
    }

    /*!
     * @brief The constructor using type DenseMat<T>.
     * @param p The dense matrix
     */
    template<class T>
    SparseMat<T>::SparseMat(DenseMat<T> &p):SparseMat(p.row(), p.col()) {
        for (int i = 1; i <= row(); i++)
            for (int j = 1; j <= col(); j++)
                set(i, j, p.get(i, j));
    }

    template<class T>
    SparseMat<T>::~SparseMat() = default;

    /*!
     *
     * @brief The same as the implementation of DenseMat.
     * @tparam T
     * @param i
     * @param j
     * @return
     */
    template<class T>
    T SparseMat<T>::get(int i, int j) {
        if (i <= 0 || j <= 0 || i > row() || j > col()) throw out_of_range("Row or column be out of range!");
        for (int i_ = 0; i_ < data_t.size(); i_++) {
            if (data_t[i_].x != i || data_t[i_].y != j) continue;
            return data_t[i_].v;
        }
        return T();
    }

    /*!
     *
     * @brief The same as the implementation of DenseMat.
     * @tparam T
     * @param i
     * @param j
     * @param v
     */
    template<class T>
    void SparseMat<T>::set(int i, int j, T v) {
        if (i <= 0 || j <= 0 || i > row() || j > col()) throw out_of_range("Row or column be out of range!");
        T zero = T();
        if (v == zero) return;
        for (int i_ = 0; i_ < data_t.size(); i_++) {
            if (data_t[i_].x == i && data_t[i_].y == j) {
                data_t[i_].v = v;
                return;
            }
        }
        data_t.push_back(triple<T>(i, j, v));
    }

    /*!
     * @brief The same as the implementation of DenseMat.
     */
    template<class T>
    SparseMat<T> &SparseMat<T>::operator=(const SparseMat<T> &p) {
        if (this == &p) return *this;

        data_t.clear();

        this->Col = p.col();
        this->Row = p.row();
        for (triple<T> e : p.data_t) data_t.push_back(e);
        return *this;
    }

    /*!
     * @brief The same as the implementation of DenseMat.
     */
    template<class T>
    void SparseMat<T>::input() {
        int x, y;
        string s1, s2, s;
        int flag = 1;
        while (true) {
            cout << "Please input the rows and columns.\n";
            cin >> s1 >> s2;
            if (s1 != to_string(stoi(s1)) || s2 != to_string(stoi(s2))) {
                cout << "The input is not integers!\n Do you want to continue? [y/n]\n";
                cin >> s;
                if (s == "y") continue;
                else {
                    flag = 0;
                    break;
                }
            }
            x = stoi(s1), y = stoi(s2);
            if (x <= 0 || x > MAX_ROW_SPARSE || y < 0 || y > MAX_COL_SPARSE) {
                cout << "rows or columns are out of range!\n Do you want to continue? [y/n]\n";
                cin >> s;
                if (s == "y") continue;
                else {
                    flag = 0;
                    break;
                }
            }
            break;
        }
        int num;
        if (!flag) return;
        while (true) {
            cout << "Please input the number of non-zero values.\n";
            cin >> s1;
            if (s1 != to_string(stoi(s1))) {
                cout << "The input is not integer!\n Do you want to continue? [y/n]\n";
                cin >> s;
                if (s == "y") continue;
                else {
                    flag = 0;
                    break;
                }
            }
            num = stoi(s1);
            if (num < 0 || num > x * y) {
                cout << "Number is out of range!\n Do you want to continue? [y/n]\n";
                cin >> s;
                if (s == "y") continue;
                else {
                    flag = 0;
                    break;
                }
            }
            break;
        }
        if (!flag) return;
        SparseMat<T> res(x, y);
        T v;

        while (true) {
            int continueFlag = 0;
            cout << "Please input n rows, each row contains 3 integer: row number, column number, value.\n";
            for (int i = 1; i <= num; i++) {
                cin >> s1 >> s2 >> v;
                if (s1 != to_string(stoi(s1)) || s2 != to_string(stoi(s2))) {
                    cout << "The input is not integers!\n Do you want to continue? [y/n]\n";
                    cin >> s;
                    if (s == "y") continueFlag = 1;
                    else flag = 0;
                    break;
                }
                x = stoi(s1), y = stoi(s2);
                res.set(x, y, v);
            }
            if (!continueFlag) break;
        }
        if (flag) *this = res;
    }
}

/*!
 * @brief The convert function from SparseMat<T> to DenseMat<T>
 * @tparam T
 * @param d a sparse matrix
 * @return a dense matrix
 */
template<class T>
dense::DenseMat<T> SparseToDense(sparse::SparseMat<T> d) {
    dense::DenseMat<T> res(d.row(), d.col());
    for (int i = 0; i < d.data_t.size(); i++)res.set(d.data_t[i].x, d.data_t[i].y, d.data_t[i].v);
    return res;
}


#endif //CPP_PROJECT_MYMATRIX_H
