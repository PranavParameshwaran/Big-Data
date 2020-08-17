#include <iostream>
#include <time.h>
struct Data
{
	int Class;
	int Freq;
};
int main()
{
	srand(time(0));

	Data d[100];
	int A[100];
	for (int i = 0; i < 100; ++i)
	{
		A[i] = rand()%10;
		d[i].Class = i+1;
		d[i].Freq = 0;
	}
	for (int i = 0; i < 100; ++i)
	{
		d[A[i]].Freq++;
	}
	char setfive[] = "||||\\";
	char one = '|';

	printf("Class\tTally\tFreq\n");
	for (int i = 0; i < 100; ++i)
	{
		if(d[i].Freq!=0)
		{
			printf("%d\t",d[i].Class);
			for(int j=0;j<(d[i].Freq/5);j++)
				printf("%s ",setfive );
			for(int j=0;j<(d[i].Freq%5);j++)
				printf("%c",one);
			printf("\t%d\n",d[i].Freq);
		}
	}

}