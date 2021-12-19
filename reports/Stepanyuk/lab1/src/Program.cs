using System;



namespace Miapr1
{
    class Program
    {
        public static void Main(string[] args)
        {
            int a = 1, b = 8;
            int inputsNum = 5;
            double d = 0.1;
            double step = 0.1;
            
            int trainingSampleSize = 30;
            int predictionRowSize = 15;
            
            double[] trainigSample = new double[trainingSampleSize];
            double[] predictionRow = new double[predictionRowSize];

            for (int i = 0; i < trainingSampleSize; i++)
            {
                trainigSample[i] = Func(step * (i + 1), a, b, d);
            }

            for (int i = 0; i < predictionRowSize; i++)
            {
                predictionRow[i] = Func(step * (trainingSampleSize + i + 1), a, b, d);
            }

            NeuralNetworkWHoff predictionMachine = new NeuralNetworkWHoff(inputsNum);
            predictionMachine.Training(trainigSample);


            Console.WriteLine("\n\n\n\nNN result   Real result   Error");

            double[] slice = new double[inputsNum];


            Array.Copy(trainigSample, 25, slice, 0, inputsNum);
            for (int i = 0; i < predictionRowSize; i++)
            {
                Console.WriteLine($"{predictionMachine.Activate(slice):f7}  {predictionRow[i]:f9}  {predictionMachine.Activate(slice) - predictionRow[i]}");

                double tmp = predictionMachine.Activate(slice);
                for (int j = 0; j < inputsNum - 1; j++)
                {
                    slice[j] = slice[j + 1];
                }
                slice[inputsNum - 1] = tmp;
            }

            predictionMachine.WriteErrorsToFile(@"errors.txt");
        }
        
        static double Func(double x, int a, int b, double d)
        {
            return a * Math.Sin(b * x) + d;
        }

    }
}



