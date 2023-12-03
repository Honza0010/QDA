#include <iostream>
#include <armadillo>
#include <vector>


#include "QDA.h"

using namespace std;
using namespace arma;


int main()
{
	/*const int N = 10;

	mat d(N, 2);
	for (int i = 0; i < N; i++)
	{

		d(i, 0) = i;
		d(i, 1) = N - i;
	}

	cout << d << endl;

	mat cov_ = cov(d);

	cout << cov(d) << endl;

	d.resize(N + 1, 3);

	cout << d << endl;

	cout << mean(d) << endl;*/




	cout << endl << endl;

	QDA q(3, 4, "iris.txt");

	cout << "predicted " << q.predict_class(std::vector<double>({5.84333333333333, 3.05733333333333, 3.75800000000000,	1.19933333333333})) << endl;


	cout << "predicted (1,1,1,1) " << q.predict_class(std::vector<double>({ 1,1,1,1 })) << endl; 

	cout << "predicted (1, 9, 2, 1) " << q.predict_class(std::vector<double>({ 1, 9, 2, 1 })) << endl;

	cout << "predicted (5, 4, 2, 0.5) " << q.predict_class(std::vector<double>({ 5, 4, 2, 0.5 })) << endl;

	//arma::mat pokus(1, 3);
	//pokus(0, 0) = 2;
	//pokus(0, 1) = 2;
	//pokus(0, 2) = 2;

	//arma::mat hokus(1, 3);
	//hokus(0, 0) = 6;
	//hokus(0, 1) = 1;
	//hokus(0, 2) = 1;

	//cout << pokus * trans(hokus) << endl;


	return 0;
}
