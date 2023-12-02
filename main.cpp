#include <iostream>
#include <armadillo>
#include <vector>


#include "QDA.h"

using namespace std;
using namespace arma;


int main()
{
	const int N = 10;
	
	mat d(N, 2);
	for (int i = 0; i < N; i++)
	{
	
		d(i, 0) = i;
		d(i, 1) = N - i;
	}

	cout << d << endl;

	mat cov_ = cov(d);

	cout << cov(d) << endl;

	d.resize(N+1, 3);

	cout << d << endl;

	cout << mean(d) << endl;




	cout << endl << endl;

	QDA q(3, 4, "iris.txt");

	q.predict_class(std::vector<double>({ 1,1,1,1 }));


	arma::mat pokus(1, 3);
	pokus(0, 0) = 2;
	pokus(0, 1) = 2;
	pokus(0, 2) = 2;

	arma::mat hokus(1, 3);
	hokus(0, 0) = 6;
	hokus(0, 1) = 1;
	hokus(0, 2) = 1;

	cout << pokus * trans(hokus) << endl;


	return 0;
}