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
	cout << cov(d) << endl;

	d.resize(N+1, 3);

	cout << d << endl;

	cout << mean(d) << endl;

	mat c(2,2);



	cout << endl << endl;

	QDA q(3, 4, "iris.txt");


	return 0;
}