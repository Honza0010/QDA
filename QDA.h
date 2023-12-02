#ifndef __QDA_
#define __QDA_

#include <vector>
#include <fstream>
#include <iostream>

#include <armadillo>

class QDA
{
	int k;	//Number of classes
	int m;	//Dimension
	std::vector<arma::mat> data;
	std::vector<arma::mat> means;

public:
	QDA(int k, int m, const std::string& filename);


private:
	void load_file(const std::string& filename);

	void calculate_means();

	void write()
	{
		std::cout << data[0] << std::endl;

		for (int i = 0; i < k; i++)
		{
			std::cout << means[i] << std::endl;
		}
	}
};


QDA::QDA(int k, int m, const std::string& filename)
	: k(k), m(m)
{
	data = std::vector<arma::mat>(k, arma::mat(0, m));
	means = std::vector<arma::mat>(k, arma::mat(1, m));

	load_file(filename);

	calculate_means();

	write();
}

void QDA::load_file(const std::string& filename)
{
	std::ifstream file;
	file.open(filename);  //otevru soubor pro zapis
	if (file.is_open())
	{
		int x;
		while (!file.eof())
		{
			
			std::vector<double> help(m + 1, 0);
			for (int i = 0; i <= this->m; i++)
			{
				file >> help[i];
			}

			int index = help[m]-1;

			arma::mat& class_matrix = data[index];
			class_matrix.resize(class_matrix.n_rows + 1, m);
			for (int i = 0; i < m; i++)
			{
				class_matrix(class_matrix.n_rows-1, i) = help[i];
			}
			//data[index].resize(data[index].n_rows + 1, m);
		}
		file.close();
	}
	else
	{

	}
}

void QDA::calculate_means()
{
	for (int i = 0; i < this->k; i++)
	{
		means[i] = arma::mean(data[i]);
	}
}

#endif // !__QDA_
