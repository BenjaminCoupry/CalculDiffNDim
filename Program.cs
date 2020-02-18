using System;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Threading.Tasks;


namespace CalculDiffNDim
{
    static class Program
    {
        /// <summary>
        /// Point d'entrée principal de l'application.
        /// </summary>
        public delegate double Application(double[] x);
        public delegate double[] FuncParam(double[] x, double[] parametres);
        public delegate double[] Transformation(double[] x);
        public enum TypeOpti {Gauss, GaussNewton,Gradient};
        public enum TypeExtremum { NonExtremum, Degenere,PointCol,Maximum,Minimum};
        [STAThread]
        static void Main()
        {
            //System.Windows.Forms.Application.EnableVisualStyles();
            //System.Windows.Forms.Application.SetCompatibleTextRenderingDefault(false);
            //System.Windows.Forms.Application.Run(new Form1());
            //x[0] latuitude, x[1] dist, x[2] altitude
            /*
             FuncParam coloriage = (x, param) =>new double[3] {param[0]*Math.Cos(x[0]*param[1]) + param[16] / (Math.Abs(param[17]) + 1 + x[1]) + param[2]*x[1]+param[3]*x[2]+param[4]+param[16]*x[1]*x[2], param[5] * Math.Cos(x[0] * param[6]) + param[18] / (Math.Abs(param[19]) + 1 + x[1]) + param[7]*x[1] + param[8] * x[2] + param[9] + param[17] * x[1] * x[2], param[10] * Math.Cos(x[0] * param[11]) + param[12] / (Math.Abs(param[13])+1 + x[1]) + param[14] * x[2] + param[15] };
             FuncParam coloriage2 = (x, param) => {
                 double glace = Heaviside(x[2]-Math.Abs(param[0]));
                 double eau = Dirac(x[1]);
                 double secheresse = Math.Exp(-Math.Pow(param[3] *(x[2] -param[4]), 2))*(1-eau);
                 double coeffVie = param[2]/ (1 + Math.Abs(param[1] * (x[2]+x[3])));
                 double r = glace+(1-glace)*(param[11]*secheresse + param[9] * eau + param[10] * (1 - eau));
                 double v = glace+(1-glace)*(param[12]*secheresse + param[7] * eau + (param[8] +coeffVie)* (1 - eau));
                 double b = glace+(1-glace)*(param[13]*secheresse + param[5]*eau + param[6] * (1 - eau));
                 return new double[3] {r,v,b};
             };
             double[] pmin = Vect1Val(-1.5,14);
             double[] pmax = Vect1Val(1.5, 14);
             InfoOpti prm = new InfoOpti(pmin,pmax,15,1,30,0.00001,0.000001,TypeOpti.GaussNewton,0.3);
             double[,][] clr = BitmapVersCarte((Bitmap)Bitmap.FromFile("C:/Users/benja/Desktop/cartes/resize/couleur.jpg"));
             double[,] alt =ApplatirCarte(BitmapVersCarte((Bitmap)Bitmap.FromFile("C:/Users/benja/Desktop/cartes/resize/altitude.jpg")));
             double[,] dstmer = ApplatirCarte(BitmapVersCarte((Bitmap)Bitmap.FromFile("C:/Users/benja/Desktop/cartes/resize/dist_mer.jpg")));
             double[,] lat = ApplatirCarte(BitmapVersCarte((Bitmap)Bitmap.FromFile("C:/Users/benja/Desktop/cartes/resize/latitude.jpg")));
             double[,] parceau = ApplatirCarte(BitmapVersCarte((Bitmap)Bitmap.FromFile("C:/Users/benja/Desktop/cartes/resize/parcEau.jpg")));
             double[,] altdist = ApplatirCarte(BitmapVersCarte((Bitmap)Bitmap.FromFile("C:/Users/benja/Desktop/cartes/resize/altfoisdist.jpg")));
             Console.WriteLine("Donnees Extraites");
             List<double[,]> listdata = new List<double[,]>();
             listdata.Add(alt);
             listdata.Add(dstmer);
             listdata.Add(lat);
             listdata.Add(parceau);
             listdata.Add(altdist);
             Tuple<double[,][], List<double[,][]>, double> cartesfinies = InspirerCarte(clr, listdata, new List<double[,][]>(), prm, coloriage2);
             (CarteVersBitmap(cartesfinies.Item1)).Save("C:/Users/benja/Desktop/cartes/resize/test0.jpg");
             Console.WriteLine(cartesfinies.Item3);
             */
            /*
           List<double[]> y = new List<double[]>();
           List<double[]> x = new List<double[]>();
           Tuple<List<double[]>, List<double[]>> xy = new Tuple<List<double[]>, List<double[]>>(x, y);
           double[] pmin = Vect1Val(-1.5, 2);
           double[] pmax = Vect1Val(1.5, 2);
           InfoOpti prm = new InfoOpti(pmin, pmax, 15, 1, 30, 0.00001, 0.000001, TypeOpti.GaussNewton, 1);
           FuncParam fp = (x_, param) => new double[1] { param[0] * x_[0] + param[1] * x_[0] * x_[1]  };
           for (int i=0;i<100;i++)
           {
               for(int j=0;j<100;j++)
               {
                   x.Add(new double[2] {i,j });
                   y.Add(new double[1] {2.6*i-i*j*3.6 });
               }
           }
           Random r = new Random();
           Tuple<double[], double, Transformation> ret = paramMinimisationErreur(xy, fp, prm, ref r);
           afficherMatrice(vectVersMat(ret.Item1));
           */
            /*
             Random r = new Random();
             double[,] M = new double[10, 10];
             for(int i=0;i<10;i++)
             {
                 for (int j = 0; j < 5; j++)
                 {
                     M[i, j] = r.Next(-9, 10);
                 }
             }
             M = sommeMat(M, transposeMat(M));
             using (System.IO.StreamWriter file =
           new System.IO.StreamWriter(@"D:/lab/errFctIt.csv"))
             {
                 for (int i = 0; i < 300; i++)
                 {
                     Console.WriteLine(i);
                     file.WriteLine(i+";"+(-Math.Log(erreurValeursPropres(vectValPropres(M, i), M))));
                 }
             }
             */
            int n1 = 30;
            double fe = 20000;
            Random r = new Random();
            int w = 20;
            double[,] messages = new double[0, 3000];
            for (int k = 0; k < w; k++)
            {
                bool[] test = new bool[n1];
                for (int i = 0; i < n1; i++)
                {
                    test[i] = r.NextDouble() > 0.5;
                }
                double[] s = BinVersAnalog(test, 0.01, fe);
                messages = ajouterLigne(messages, s);
            }
            double[] sg =BruiterSignal(Transmettre(messages, 100,fe),20,fe,ref r);
            enregistrerMatrice(transposeMat(messages), "D:/lab/messagesOrigine.csv");
            enregistrerMatrice(vectVersMat(DSP(sg,nexpPow2(sg.Length))), "D:/lab/dsp.csv");
            double[,] recu = Recevoir(sg, w, 100, fe, false, 0.01);
            enregistrerMatrice(transposeMat(recu), "D:/lab/messagesRecu.csv");
           
            enregistrerMatrice(transposeMat(sommeMat(recu,scalaireMat(-1,messages))), "D:/lab/diffMessage.csv");
        }


        //Fonction a parametres specifiques
        public static double CombLin(double[] param, int k, double[] x)
        {
            //Retourne la combinaison linéaire des param[k+i]*x[i]
            double res = 0;
            for (int i = 0; i < x.Length; i++)
            {
                res += param[k] * x[i];
                k++;
            }
            return res;
        }
        public static double Polynome(double[] param, int k,int n, double x)
        {
            //Retourne le polynome de degré n pris en x avec comme coeff les param[k+i]
            double res = 0;
            for (int i = 0; i < n; i++)
            {
                res += param[k] * Math.Pow(x,i);
                k++;
            }
            return res;
        }
        public static double Polynome(double[] param, int k, int n, double[] x)
        {
            //Retourne la somme des polynomes de degré n pris en xi avec comme coeff les param[k+j]
            double res = 0;
            int nbelt = x.GetLength(0);
            int k0 = k;
            for (int i = 0; i <nbelt ; i++)
            {
                res += Polynome(param, k0, n, x[i]);
                k0 += n;
            }
            return res;
        }
        public static double[] TermeATerme(double[] x, Application f)
        {
            //Applique f a chaque terme de x
            double[] res = new double[x.Length];
            for(int i=0;i<x.Length;i++)
            {
                res[i] = f(new double[1] { x[i] });
            }
            return res;
        }
        public static double[] TermeATerme(double[] x, List<Application> f)
        {
            //Applique les elements de f a chaque terme de x, puis concatène les resultats
            double[] res = new double[x.Length*f.Count];
            int n = 0;
            for (int i = 0; i < x.Length; i++)
            {
                for(int k=0;k<f.Count;k++)
                {
                    res[n] = f.ElementAt(k)(new double[1] { x[i] });
                    n++;
                }
            }
            return res;
        }
        public static double[] Vect1Val(double val, int n)
        {
            //Applique f a chaque terme de x
            double[] res = new double[n];
            for (int i = 0; i < n; i++)
            {
                res[i] = val;
            }
            return res;
        }
        public static double Heaviside(double x)
        {
            if(x>0)
            {
                return 1;
            }
            else if(x<0)
            {
                return 0;
            }
            else
            {
                return 0.5;
            }
        }
        public static double Dirac(double x)
        {
            if (x == 0)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }
        public static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        //Derivees
        public static double[] gradient(double[] x, double delta, Application f)
        {
            //Retourne le gradient de f au point x, avec un delta donné
            int Taille = x.Length;
            double[] retour = new double[Taille];
            double y1 = f(x);
            for (int i = 0; i < Taille; i++)
            {
                x[i] += delta;
                double y2 = f(x);
                x[i] -= delta;
                retour[i] = (y2 - y1) / delta;
            }
            return retour;
        }
        public static double[,] jacobienne(double[] x, double delta, Transformation f)
        {
            //Retourne la jacobienne de f au point x, avec un delta donné
            int Tx = x.Length;
            double[] f0 = f(x);
            int Ty = f0.Length;
            double[,] resultat = new double[Ty, Tx];
            for (int i = 0; i < Tx; i++)
            {
                Console.WriteLine("[" + i + "/" + Tx + "]");
                x[i] += delta;
                double[] f1 = f(x);
                x[i] -= delta;
                for (int j = 0; j < Ty; j++)
                {
                    resultat[j, i] = (f1[j] - f0[j]) / delta;
                }
            }
            return resultat;
        }
        public static double[,] hessienne(double[] x, double delta, Application f)
        {
            // Retourne la hessienne de f au point x, avec un delta donné
            int Taille = x.Length;
            Transformation grad = fparam => gradient(fparam, delta, f);
            double[,] retour = jacobienne(x, delta, grad);
            //La matrice doit etre symetrique
            return scalaireMat(0.5, sommeMat(retour, transposeMat(retour)));
        }
        public static double hessien(double[] x, double delta, Application f)
        {
            //Renvoie le hessien de f au point x
            return det(hessienne(x, delta, f));
        }
        public static double divergence(double[] x, double delta, Transformation f)
        {
            //retourne la divergence de f au point x
            double[,] J = jacobienne(x, delta, f);
            return trace(J);
        }
        public static double laplacien(double[] x, double delta, Application f)
        {
            //Retourne le laplacien de f en x
            double[,] H = hessienne(x, delta, f);
            return trace(H);
        }

        //Erreur
        public static double[] vecteurResidu(List<double[]> x_ech, List<double[]> y_ech, double[] parametres, FuncParam f_estimation)
        {
            int n = Math.Min(x_ech.Count, y_ech.Count);
            double[] result = new double[n*y_ech.ElementAt(0).Length];
            int k = 0;
            for (int i = 0; i < n; i++)
            {
                double[] estim = f_estimation(x_ech.ElementAt(i), parametres);
                for(int z=0;z<estim.Length;z++)
                {
                    result[k] = y_ech.ElementAt(i)[z] - estim[z];
                    k++;
                }
            }
            return result;
        }
        public static double erreurEstimation(List<double[]> x_ech, List<double[]> y_ech, double[] parametres, FuncParam f_estimation)
        {
            //Donne l'erreur d'estimation du set x_param y_param par la fonction f_estimation munie des parametres parametres
            double[] residu = vecteurResidu(x_ech,  y_ech, parametres, f_estimation);
            return Math.Pow(normeVect(residu), 2)/residu.Length;
        }
        public static Tuple<double[], double, Transformation> paramMinimisationErreur(Tuple<List<double[]>, List<double[]>> xech_yech, FuncParam f_estimation, InfoOpti paramOpti, ref Random r)
        {
            //Renvoie les parametres qui minimisent l'erreur d'estimation du set x_ech_yech par la fonction f_estimation, munie initialement de paraminit compris entre min et max
            //Ainsi que la fonction qui prend les parametres optimaux, et l'erreur de cette fonction
            double[] param_min = paramOpti.param_min;
            double[] param_max = paramOpti.param_max;
            int nbStagnationsMax = paramOpti.nbStagnationsMax;
            int nbDepartsMax = paramOpti.nbDepartsMax;
            int iterMax_newton = paramOpti.iterMax_newton;
            double relStop_newton = paramOpti.relStop_newton;
            double delta_diff = paramOpti.delta_diff;
            List<double[]> x_ech = xech_yech.Item1;
            List<double[]> y_ech = xech_yech.Item2;
            Application err = fparam => erreurEstimation(x_ech, y_ech, fparam, f_estimation);
            Transformation res = fparam => vecteurResidu(x_ech, y_ech, fparam, f_estimation);
            double[] paramOptimaux = new double[param_min.Length];
            double errMin = double.PositiveInfinity;
            int n = 0;
            int N = 0;
            Tuple<double[], double> Optimaux_Locaux;
            do
            {
                n++;
                double[] paramTest_init = vectAlea(ref r, param_min, param_max);
                if(paramOpti.typeOpti == TypeOpti.Gauss)
                {
                    Optimaux_Locaux = minimiser_N(err, paramTest_init, iterMax_newton, relStop_newton, delta_diff,paramOpti.pasGrad);
                }
                else if(paramOpti.typeOpti == TypeOpti.GaussNewton)
                {
                    Optimaux_Locaux = minimiser_GN(res, paramTest_init, iterMax_newton, relStop_newton, delta_diff,paramOpti.pasGrad);
                }
                else
                {
                    Optimaux_Locaux = minimiser_Grad(err, paramTest_init, iterMax_newton, relStop_newton, delta_diff,paramOpti.pasGrad);
                }
                
                if (Optimaux_Locaux.Item2 < errMin)
                {
                    paramOptimaux = Optimaux_Locaux.Item1;
                    errMin = Optimaux_Locaux.Item2;
                    N = 0;
                }
                else
                {
                    N++;
                }
                Console.WriteLine("------------------");
                Console.WriteLine(n+"/"+nbDepartsMax+" || "+N+"/"+nbStagnationsMax);
                Console.WriteLine("------------------");
            }
            while (n < nbDepartsMax && N < nbStagnationsMax);

            Transformation TransformationOptimale = x_tr => f_estimation(x_tr, paramOptimaux);
            Console.WriteLine("Resultat :");
            afficherMatrice(vectVersMat(paramOptimaux));
            return new Tuple<double[], double, Transformation>(paramOptimaux, errMin, TransformationOptimale);
        }

        //Matrices
        public static double[] ligne(double[,] M,int i)
        {
            //Recupere la ligne i de M
            int k = M.GetLength(1);
            double[] ret = new double[k];
            for(int l=0;l<k;l++)
            {
                ret[l] = M[i, l];
            }
            return ret;
        }
        public static double[,] ajouterLigne(double[,] M, double[]ligne)
        {
            //Ajoute une ligne a la fin de M
            int n = M.GetLength(0);
            int k = M.GetLength(1);
            double[,] res = new double[n+1, k];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    res[i, j] = M[i, j];
                }
            }
            for(int j=0;j<k;j++)
            {
                res[n, j] = ligne[j];
            }
            return res;
        }
        public static double[,] identite(int n)
        {
            double[,] res = new double[n, n];
            for(int i=0;i<n;i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if(i==j)
                    {
                        res[i, j] = 1;
                    }
                    else
                    {
                        res[i, j] = 0;
                    }
                }
            }
            return res;
        }
        public static double[,] copieMat(double[,]M)
        {
            int n = M.GetLength(0);
            int k = M.GetLength(1);
            double[,] res = new double[n, k];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    res[i, j] = M[i, j];
                }
            }
            return res;
        }
        public static double[,] vectVersMat(double[] v)
        {
            //transforme un vecteur en matrice
            double[,] retour = new double[v.Length, 1];
            for (int i = 0; i < v.Length; i++)
            {
                retour[i, 0] = v[i];
            }
            return retour;
        }
        public static double[] matVersVect(double[,] M)
        {
            //transforme une matrice en vecteur
            double[] retour = new double[M.GetLength(0)];
            for (int i = 0; i < M.GetLength(0); i++)
            {
                retour[i] = M[i, 0];
            }
            return retour;
        }
        public static double[,] scalaireMat(double k, double[,] M)
        {
            //Multiplie la matrice par un scalaire
            double[,] retour = new double[M.GetLength(0), M.GetLength(1)];
            for (int i = 0; i < M.GetLength(0); i++)
            {
                for (int j = 0; j < M.GetLength(1); j++)
                {
                    retour[i, j] = k * M[i, j];
                }
            }
            return retour;
        }
        public static double[,] produitMat(double[,] A, double[,] B)
        {
            //Produit matriciel
            double[,] retour = new double[A.GetLength(0), B.GetLength(1)];
            for (int i = 0; i < A.GetLength(0); i++)
            {
                for (int j = 0; j < B.GetLength(1); j++)
                {
                    double somme = 0;
                    for (int k = 0; k < A.GetLength(1); k++)
                    {
                        somme += A[i, k] * B[k, j];
                    }
                    retour[i, j] = somme;
                }
            }
            return retour;
        }
        public static double[,] sommeMat(double[,] A, double[,] B)
        {
            //Somme des matrices
            double[,] retour = new double[A.GetLength(0), A.GetLength(1)];
            for (int i = 0; i < A.GetLength(0); i++)
            {
                for (int j = 0; j < A.GetLength(1); j++)
                {
                    retour[i, j] = A[i, j] + B[i, j];
                }
            }
            return retour;
        }
        public static double[,] transposeMat(double[,] M)
        {
            //Matrice transposee
            double[,] retour = new double[M.GetLength(1), M.GetLength(0)];
            for (int i = 0; i < M.GetLength(1); i++)
            {
                for (int j = 0; j < M.GetLength(0); j++)
                {
                    retour[i, j] = M[j, i];
                }
            }
            return retour;
        }
        public static double[,] matriceTronquee(double[,] M, int i0, int j0)
        {
            //Retourne la matrice M privee de sa ieme ligne et jemme colonne
            double[,] retour = new double[M.GetLength(0) - 1, M.GetLength(1) - 1];
            for (int i = 0; i < M.GetLength(0); i++)
            {
                for (int j = 0; j < M.GetLength(1); j++)
                {
                    int j_eff;
                    int i_eff;
                    if (j > j0)
                    {
                        j_eff = j - 1;
                    }
                    else
                    {
                        j_eff = j;
                    }
                    if (i > i0)
                    {
                        i_eff = i - 1;
                    }
                    else
                    {
                        i_eff = i;
                    }
                    if ((j != j0) && (i != i0))
                    {
                        retour[i_eff, j_eff] = M[i, j];
                    }
                }
            }
            return retour;
        }
        public static double det(double[,] M)
        {
            //Determinant d'une matrice
            int n = M.GetLength(0);

            if (n == 1)
            {
                return M[0, 0];
            }
            else
            {
                double[,] M_Travail = new double[n - 1, n - 1];
                int N0 = 0;
                while (N0 < n && M[0, N0] == 0)
                {
                    //Chercher un element non nul sur la premiere ligne
                    N0++;
                }
                if (N0 == n)
                {
                    //La premiere ligne est composee de 0
                    return 0;
                }
                else
                {
                    //Annuler tous les elements autres que N0 de la premiere ligne
                    for (int i = 0; i < n - 1; i++)
                    {
                        for (int j = 0; j < n; j++)
                        {
                            if (j != N0)
                            {
                                int jeff;
                                if (j > N0)
                                {
                                    jeff = j - 1;
                                }
                                else
                                {
                                    jeff = j;
                                }
                                double k = -M[0, j] / M[0, N0];
                                M_Travail[i, jeff] = M[i + 1, j] + k * M[i + 1, N0];
                            }
                        }
                    }
                    //Developper par rapport a la premiere ligne, et donc par rapport a l element N0
                    return M[0, N0] * det(M_Travail) * Math.Pow(-1, N0);
                }
            }
        }
        public static double[,] comatrice(double[,] M)
        {
            //Comatrice de M
            int n = M.GetLength(0);
            double[,] resultat = new double[n, n];
            double sgn = 1;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    sgn = Math.Pow(-1, i + j);
                    resultat[i, j] = sgn * det(matriceTronquee(M, i, j));
                }
            }
            return resultat;
        }
        public static double[,] inverse(double[,] M)
        {
            //Inverse la matrice M
            if (M.GetLength(0) == 1)
            {
                double[,] res = new double[1, 1];
                res[0, 0] = 1.0 / M[0, 0];
                return res;
            }
            else
            {
                return scalaireMat(1.0 / det(M), transposeMat(comatrice(M)));
            }
        }
        public static double trace(double[,] M)
        {
            //Retourne la trace d'une matrice
            double result = 0;
            for (int i = 0; i < M.GetLength(0); i++)
            {
                result += M[i, i];
            }
            return result;
        }
        public static void afficherMatrice(double[,] M)
        {
            //Affiche une matrice
            for (int i = 0; i < M.GetLength(0); i++)
            {
                string s = "";
                for (int j = 0; j < M.GetLength(1); j++)
                {
                    s = s + M[i, j].ToString("0.00") + "  ";
                }
                Console.WriteLine("");
                Console.WriteLine(s);
            }
        }
        public static void enregistrerMatrice(double[,] M, string Path)
        {
            //Enregistre une matrice
            using (System.IO.StreamWriter file =
           new System.IO.StreamWriter(@Path))
            {
                for (int i = 0; i < M.GetLength(0); i++)
                {
                    string s = "";
                    for (int j = 0; j < M.GetLength(1); j++)
                    {
                        s = s + M[i, j].ToString() + ";";
                    }
                    file.WriteLine(s);
                }
            }
        }
        public static double[,] completerMatrice(double[,] M,int n)
        {
            //Complete une matrice avec des 1 sur la diagonale gauche jusqu a la taille n
            double[,] result = identite(n);
            int n0 = n - M.GetLength(0);
            for(int i=n0;i<n;i++)
            {
                for (int j = n0; j < n; j++)
                {
                    result[i, j] = M[i - n0, j - n0];
                }
            }
            return result;
        }
        private static Tuple<double[,], double[,]> decompQR(double[,]M)
        {
            //decompose M en QR avec Q orthogonale et R triangulaire sup
            int n = M.GetLength(0);
            if (n == 1)
            {
                return new Tuple<double[,], double[,]> (identite(1),identite(1));
            }
            else
            {
                double[] x = new double[n];
                for (int i = 0; i < n; i++)
                {
                    x[i] = M[i, 0];
                }
                double sg;
                if(x[0]==0)
                {
                    sg = 1;
                }
                else
                {
                    sg = Math.Sign(x[0]);
                }
                double alpha = -normeVect(x) * sg;
                x[0] -= alpha;
                multiplierVect(ref x, 1.0 / normeVect(x));
                double[,] Q1 = sommeMat(identite(n), scalaireMat(-2, produitMat(vectVersMat(x), transposeMat(vectVersMat(x)))));
                double[,] Q1M = produitMat(Q1, M);
                Tuple<double[,], double[,]> Qsuiv = decompQR(matriceTronquee(Q1M, 0, 0));
                double[,] Q = produitMat(Q1, completerMatrice(Qsuiv.Item1, n));
                double[,] R = produitMat(transposeMat(Q), M);
                return new Tuple<double[,], double[,]>(Q, R);
            }
        }
        public static double[,] QRHessenbergTrig(double[,] M, int nbIt)
        {
            //Cherche une matrice triangulaire semblable à M en nbIt iterations
            double[,] H = copieMat(M);
            for(int i=0;i<nbIt;i++)
            {
                Tuple<double[,], double[,]> QR = decompQR(H);
                H = produitMat(QR.Item2, QR.Item1);
            }
            return H;
        }
        public static double[] diagonale(double[,] M)
        {
            double[] retour = new double[M.GetLength(0)];
            for(int i=0;i<M.GetLength(0);i++)
            {
                retour[i] = (M[i, i]);
            }
            return retour;
        }
        public static double[,] echangerLignes(double[,] M, int i,int j)
        {
            //Echange la ligne i et j de la matrice
            double[,] resultat = copieMat(M);
            int n = M.GetLength(1);
            for(int k=0;k<n;k++)
            {
                resultat[i, k] = M[j, k];
                resultat[j, k] = M[i, k];
            }
            return resultat;
        }
        public static double[,] eliminationGauss(double[,] M)
        {
            double[,] Mat = copieMat(M);
            int m = M.GetLength(0);
            int n = M.GetLength(1);
            int h = 0;
            int k = 0;
            while(h<m && k<n)
            {
                //Trouver imax
                double vmax = 0;
                int imax=0;
                for(int z=h;z<m;z++)
                {
                    if(Math.Abs(Mat[z,k])>=vmax)
                    {
                        imax = z;
                        vmax = Math.Abs(Mat[z, k]);
                    }
                }
                if(Mat[imax,k]==0)
                {
                    k++;
                }
                else
                {
                    Mat = echangerLignes(Mat, h, imax);
                    for(int i=h+1;i<m;i++)
                    {
                        double f = Mat[i, k] / Mat[h, k];
                        Mat[i, k] = 0;
                        for(int j = k+1;j<n;j++)
                        {
                            Mat[i, j] = Mat[i, j] - Mat[h, j] * f;
                        }
                    }
                    h++;
                    k++;
                }
            }
            return Mat;
        }
        public static double[] vectKer(double[,]M)
        {
            //Retourne un vecteur non nul normé du ker de M si il est de dim 1
            int n = M.GetLength(0);
            double[,] MG = eliminationGauss(M);
            double[] result = new double[n];
            result[n - 1] = 1;
            for(int i=n-2;i>=0;i--)
            {
                double somme = 0;
                for (int j = n - 1; j >i; j--)
                {
                    somme -= MG[i,j]*result[j];
                }
                result[i] = somme / MG[i, i];
            }
            multiplierVect(ref result, 1.0 / normeVect(result));
            return result;
        }
        public static double[] valeursPropres(double[,] M, int nbIt)
        {
            //Retourne les valeurs propres de M si M est diagonalisable
            return diagonale(QRHessenbergTrig(M, nbIt));
        }
        public static List<double[]> vecteursPropres(double[,] M, int nbIt)
        {
            //Renvoie une famille de vecteurs propres de M
            List<double[]> result = new List<double[]>(); ;
            List<Tuple<double[], double>> vvp = vectValPropres(M, nbIt);
            for(int i=0;i<vvp.Count;i++)
            {
                result.Add(vvp.ElementAt(i).Item1);
            }
            return result;
        }
        public static List<double[]> OrthogonaliserGS(List<double[]> famille)
        {
            //Orthogonalise la famille vecteurs selon graam schmidt
            List<double[]> ui = new List<double[]>();
            for (int k=0;k<famille.Count;k++)
            {
                double[] somme =copiedb(famille.ElementAt(k));
                for(int j=0;j<k;j++)
                {
                    double[] prj = projeter(ui.ElementAt(j), famille.ElementAt(k));
                    multiplierVect(ref prj,-1);
                    somme = sommeVect(somme, prj);
                }
                ui.Add(normerVect(somme));
            }
            return ui;
        }
        public static double erreurValeursPropres(List<Tuple<double[], double>> valVectPropres, double[,] M)
        {
            //Evalue l'erreur d'une liste de couples de vecteurs et valeurs propres de M
            double err = 0;
            for (int i = 0; i < valVectPropres.Count; i++)
            {
                double val = valVectPropres.ElementAt(i).Item2;
                double[] v = valVectPropres.ElementAt(i).Item1;
                err+=Math.Pow(normeVect(matVersVect(sommeMat(vectVersMat(v), scalaireMat(-1.0 / val, produitMat(M, vectVersMat(v)))))),2);
            }
            err = Math.Sqrt(err);
            return err;
        }
        public static List<Tuple<double[],double>> vectValPropres(double[,]M,int itQR)
        {
            //Retourne liste de couples vecteurs/valeurs propres de M
            double[] vp = valeursPropres(M, itQR);
            List<Tuple<double[], double>> retour = new List<Tuple<double[], double>>();
            for (int i=0;i<vp.Length;i++)
            {
                double[,] Mr = sommeMat(M,scalaireMat(-vp[i],identite(vp.Length)));
                double[] Vect = vectKer(Mr);
                retour.Add(new Tuple<double[], double>(Vect, vp[i]));
            }
            return retour;
        }
        public static Tuple<double[],double> rechercherVectPropre(double[,] M,double[] vp0, int n)
        {
            //Iteration du quotient de Rayleigh, retourne un vecteur propre et sa valeur propre associee
            double[] vect = copiedb(vp0);
            double lambda = 0;
            for(int i=0;i<n;i++)
            {
                lambda = produitScalaire(vect,matVersVect(produitMat(M,vectVersMat(vect)))) / produitScalaire(vect,vect);
                double[] k = matVersVect(produitMat(inverse(sommeMat(M, scalaireMat(-lambda, identite(vect.Length)))), vectVersMat(vect)));
                multiplierVect(ref k, 1.0 / normeVect(k));
                vect = copiedb(k);
            }
            lambda = produitScalaire(vect, matVersVect(produitMat(M, vectVersMat(vect)))) / produitScalaire(vect, vect);
            return new Tuple<double[], double>(vect, lambda);
        }


        //Optimisation
        public static TypeExtremum caracteriserPoint(Application f, double[] x0, double delta, double epsilon,int nbIt)
        {
            double[] grad = gradient(x0, delta, f);
            if (normeVect(grad) > epsilon)
            {
                return TypeExtremum.NonExtremum;
            }
            else
            {
                double[,] H = hessienne(x0, delta, f);
                if(Math.Abs(det(H))<=epsilon)
                {
                    return TypeExtremum.Degenere;
                }
                else
                {
                    double[] vp = valeursPropres(H, nbIt);
                    int n = vp.Length;
                    bool Pos=true;
                    bool Neg=true;
                    for(int i=0;i<n;i++)
                    {
                        Pos = Pos && vp[i] > 0;
                        Neg = Neg && vp[i] < 0;
                    }
                    if(Pos)
                    {
                        return TypeExtremum.Minimum;
                    }
                    else if(Neg)
                    {
                        return TypeExtremum.Maximum;
                    }
                    else
                    {
                        return TypeExtremum.PointCol;
                    }
                }
            }
        }
        //Algo newton
        public static Tuple<double[], double> minimiser_N(Application f, double[] x0, int iterMax, double relStop, double delta,double pasGrad)
        {
            //annule grad(f) par algorithme de newton en partant de x0, et renvoie la valeur de la fonction au point x final
            int nb_iter = 0;
            double[] x = copiedb(x0);
            double[] grad = gradient(x0, delta, f);
            double nrmgrad0 = normeVect(grad);
            double[,] hess;
            double[] pas;
            double Rel_Erreur_g;
            double val;
            do
            {
                Console.WriteLine("iter :" + nb_iter);
                Console.WriteLine("Calcul de la hessienne");
                hess = hessienne(x, delta, f);
                Console.WriteLine("det =" + det(hess));
                if (det(hess) != 0)
                {
                    Console.WriteLine("Inversion de la hessienne");
                    double[,] hessInv = inverse(hess);
                    pas = matVersVect(scalaireMat(-pasGrad, produitMat(hessInv, vectVersMat(grad))));
                }
                else
                {
                    Console.WriteLine("Suivre le gradient");
                    multiplierVect(ref grad, -pasGrad);
                    pas = grad;
                }
                Console.WriteLine("Application du pas");
                x = sommeVect(x, pas);
                nb_iter++;
                Console.WriteLine("Calcul du gradient");
                grad = gradient(x, delta, f);
                Rel_Erreur_g = normeVect(grad) / (nrmgrad0 + 1);
                Console.WriteLine("norme relative du gradient : " + Rel_Erreur_g);
                val = f(x);
                Console.WriteLine("Valeur de la fonction : " + val);
            }
            while (nb_iter < iterMax && Rel_Erreur_g > relStop);
            return new Tuple<double[], double>(x, val);
        }
        //Algo Gauss Newton
        public static Tuple<double[], double> minimiser_GN(Transformation r, double[] x0, int iterMax, double relStop, double delta,double pasGrad)
        {
            //Minimise la norme au carré de r, partant de x0.
            int nb_iter = 0;
            double[] x = copiedb(x0);
            double val0 = normeVect(r(x0));
            double[] pas;
            double[,] jacob;
            double[,] jacobT;
            double val;
            do
            {
                Console.WriteLine("iter :" + nb_iter);
                //Calcul du pas
                //Jacobienne des residus
                Console.WriteLine("Calcul de la Jacobienne");
                jacob = jacobienne(x, delta, r);
                Console.WriteLine("Transposition");
                jacobT = transposeMat(jacob);
                Console.WriteLine("Inversion");
                double[,] inv = inverse(produitMat(jacobT, jacob));
                Console.WriteLine("Calcul du pas");
                pas =matVersVect(scalaireMat(-pasGrad, produitMat(inv,produitMat(jacobT,vectVersMat(r(x))))));
                Console.WriteLine("Application du pas");
                x = sommeVect(x, pas);
                nb_iter++;
                val = Math.Pow(normeVect(r(x)),2);
                Console.WriteLine("Valeur de la fonction : " + val);
            }
            while (nb_iter < iterMax && val/(1.0+val0) > relStop);
            return new Tuple<double[], double>(x, val);
        }
        //Algo gradient
        public static Tuple<double[], double> minimiser_Grad(Application f, double[] x0, int iterMax, double relStop, double delta,double pasGrad)
        {
            //annule grad(f) par algorithme de newton en partant de x0, et renvoie la valeur de la fonction au point x final
            int nb_iter = 0;
            double[] x = copiedb(x0);
            double[] grad = gradient(x0, delta, f);
            double nrmgrad0 = normeVect(grad);
            double[] pas;
            double Rel_Erreur_g;
            double val;
            do
            {
                Console.WriteLine("iter :" + nb_iter);
                Console.WriteLine("Suivre le gradient");
                multiplierVect(ref grad, -pasGrad);
                pas = grad;
                Console.WriteLine("Application du pas");
                x = sommeVect(x, pas);
                nb_iter++;
                Console.WriteLine("Calcul du gradient");
                grad = gradient(x, delta, f);
                Rel_Erreur_g = normeVect(grad) / (nrmgrad0 + 1);
                Console.WriteLine("norme relative du gradient : " + Rel_Erreur_g);
                val = f(x);
                Console.WriteLine("Valeur de la fonction : " + val);
            }
            while (nb_iter < iterMax && Rel_Erreur_g > relStop);
            return new Tuple<double[], double>(x, val);
        }
        //Discretisation
        private static void CoordPaveRecursif(double[] xmin, double[] xmax, double[] deltasX, int n, ref List<double[]> coords, double[] coor)
        {
            //Utilise pour coordPave
            int nbpas = (int)((xmax[n] - xmin[n]) / deltasX[n]);
            for (int i = 0; i <= nbpas; i++)
            {
                double[] coordNext = copiedb(coor);
                coordNext[n] = xmin[n] + i * deltasX[n];
                coordNext = copiedb(coordNext);
                if (n == 0)
                {
                    coords.Add(coordNext);
                }
                else
                {
                    CoordPaveRecursif(xmin, xmax, deltasX, n - 1, ref coords, coordNext);
                }
            }
        }
        public static List<double[]> CoordPave(double[] xmin, double[] xmax, double[] deltasX)
        {
            //Renvoie la liste des coordonées du pavé entre xmin et xmax, choisies avec un pas de deltasX[n] pour la nieme dim
            List<double[]> retour = new List<double[]>();
            CoordPaveRecursif(xmin, xmax, deltasX, Math.Min(xmin.Length, xmax.Length) - 1, ref retour, new double[Math.Min(xmin.Length, xmax.Length)]);
            return retour;
        }
        public static int[] Placer(double[] ymin, double[] ymax, int[] subdivY, double[] x)
        {
            //Placer x dans une grille construite dans le pave ymin ymax, avec subdivY[n] cases pour la nieme dim
            int n = Math.Min(Math.Min(x.Length, subdivY.Length), Math.Min(ymax.Length, ymin.Length));
            int[] coords = new int[n];
            for (int i = 0; i < n; i++)
            {
                double amplitude = ymax[i] - ymin[i];
                coords[i] = (int)((Math.Min(1.0, Math.Max(0.0, (x[i] - ymin[i]) / amplitude))) * (double)(subdivY[i] - 1));
            }
            return coords;
        }
        public static NdimArray CarteIntensite(double[] xmin, double[] xmax, double[] deltasX, double[] ymin, double[] ymax, int[] subdivY, Transformation tr)
        {
            //Renvoie une carte des intensités (probabilités) des images du pavé compris entre xmin et xmax, pour les images inclues dans le pave ymin, ymax.
            //Choix des abscisses choisies avec un pas de deltasX[n] pour la nieme dim.
            //subdivY est le nombre de cases a faire dans chaque dimention de l'image
            List<double[]> coords = CoordPave(xmin, xmax, deltasX);
            NdimArray retour = new NdimArray(subdivY);
            retour.zeros();
            for (int i = 0; i < coords.Count; i++)
            {
                retour.add(Placer(ymin, ymax, subdivY, tr(coords.ElementAt(i))), 1.0);
            }
            retour.normaliser();
            return retour;
        }

        //Cartes 2D
        public static double[,][] BitmapVersCarte(Bitmap image)
        {
            //Transforme une image a 3 caneaux en une grille de vecteurs

            int x = image.Width;
            int y = image.Height;
            double[,][] carte = new double[x, y][];
            for(int i=0;i<x;i++)
            {
                for (int j = 0; j < y; j++)
                {
                    Color pix = image.GetPixel(i, j);
                    double[] pix_ = new double[3] {pix.R/255.0,pix.G/255.0,pix.B/255.0};
                    carte[i, j] = pix_;
                }
            }
            return carte;
        }
        public static Bitmap CarteVersBitmap(double[,][] carte)
        {
            //Transforme une carte a 3 canaux en bitmap
            int x = carte.GetLength(0);
            int y = carte.GetLength(1);
            Bitmap retour = new Bitmap(x, y);
            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    Color pix = Color.FromArgb((int)(Math.Min(1,Math.Max(carte[i,j][0],0))*255), (int)(Math.Min(1, Math.Max(carte[i, j][1], 0)) * 255), (int)(Math.Min(1, Math.Max(carte[i, j][2], 0)) * 255));
                    retour.SetPixel(i, j, pix);
                }
            }
            return retour;

        }
        public static double[,] ApplatirCarte(double[,][] carte)
        {
            //Applatit une carte de vecteurs en carte de double
            int x = carte.GetLength(0);
            int y = carte.GetLength(1);
            double[,] retour = new double[x, y];
            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    retour[i, j] = carte[i, j].Sum() / carte[i, j].GetLength(0);
                }
            }
            return retour;
        }
        public static double[,][] GonflerCarte(double[,] carte,int n)
        {
            // transforme une carte de double en carte de vecteurs (vecteurs de toutes coord egales)
            int x = carte.GetLength(0);
            int y = carte.GetLength(1);
            double[,][] retour = new double[x, y][];
            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    double val = carte[i, j];
                    double[] vect = new double[n];
                    for(int k =0;k<n;k++)
                    {
                        vect[k] = val;
                    }
                    retour[i, j] =vect;
                }
            }
            return retour;
        }
        public static double[,][] ImagesPlacees(double[,][] abscisses, Transformation f)
        {

            //Renvoie l'image terme a terme d'une grille 2D de vecteurs par la transformation f
            int X = abscisses.GetLength(0);
            int Y = abscisses.GetLength(1);
            double[,][] retour = new double[X, Y][];
            for (int i = 0; i < X; i++)
            {
                for (int j = 0; j < Y; j++)
                {
                    double[] y = copiedb(f(abscisses[i, j]));
                    retour[i, j] = y;
                }
            }
            return retour;
        }
        public static Tuple<List<double[]>, List<double[]>> Echantillonage(double[,][] mapX, double[,][] mapY)
        {
            //Liste les couples (X,Y) pour les grilles mapX et mapY
            List<double[]> x = new List<double[]>();
            List<double[]> y = new List<double[]>();
            int X = mapX.GetLength(0);
            int Y = mapX.GetLength(1);
            for (int i = 0; i < X; i++)
            {
                for (int j = 0; j < Y; j++)
                {
                    double[] xi = copiedb(mapX[i, j]);
                    double[] yi = copiedb(mapY[i, j]);
                    x.Add(xi);
                    y.Add(yi);
                }
            }
            return new Tuple<List<double[]>, List<double[]>>(x, y);
        }
        public static double[,][] RassemblerCartes(List<double[,]> cartes)
        {
            //rassemble un ensemble de cartes de doubles en une carte de vecteurs
            int X = cartes.ElementAt(0).GetLength(0);
            int Y = cartes.ElementAt(0).GetLength(1);
            double[,][] retour = new double[X, Y][];
            int nbCartes = cartes.Count;
            for (int i = 0; i < X; i++)
            {
                for (int j = 0; j < Y; j++)
                {
                    double[] val = new double[nbCartes];
                    for (int k = 0; k < nbCartes; k++)
                    {
                        val[k] = cartes.ElementAt(k)[i, j];
                    }
                    retour[i, j] = val;
                }
            }
            return retour;
        }
        public static List<double[,]> EclaterCarte(double[,][] carte)
        {
            //Eclate une carte de vecteurs en une liste de carte de double (inverse de RassemblerCartes)
            List<double[,]> retour = new List<double[,]>();
            int X = carte.GetLength(0);
            int Y = carte.GetLength(1);
            int nbdim = carte[0, 0].Length;
            for (int k = 0; k < nbdim; k++)
            {
                double[,] cartei = new double[X, Y];
                for (int i = 0; i < X; i++)
                {
                    for (int j = 0; j < Y; j++)
                    {
                        cartei[i, j] = carte[i, j][k];
                    }
                }
                retour.Add(cartei);
            }
            return retour;
        }
        public static Tuple<double[,][], List<double[,][]>, double> InspirerCarte(double[,][] RepresentationAttendue, List<double[,]> CartesDonnees, List<double[,][]> DonneesARepresenter, InfoOpti paramOptimisation, FuncParam fonctionRepresentation)
        {
            //Optimise la fonction fonctionRepresentation avec les parametres paramOptimisation pour faire concorder les donnees cartesDonnees avec la representationAttndue.
            //La fonction optimisee est alors apliquuée sur la liste de cartes DonneesARepresenter
            //Retour : les donnees initiales represntees par la fonction, la liste des representations voulues, et l'erreur d'optimisation
            Random r = new Random();
            double[,][] carteDonnees = RassemblerCartes(CartesDonnees);
            Tuple<double[], double, Transformation> resultOpti = paramMinimisationErreur(Echantillonage(carteDonnees, RepresentationAttendue), fonctionRepresentation, paramOptimisation, ref r);
            Transformation representation = resultOpti.Item3;
            double[,][] carteInitRepresentee = ImagesPlacees(carteDonnees, representation);
            List<double[,][]> representees = new List<double[,][]>();
            for (int i = 0; i < DonneesARepresenter.Count; i++)
            {
                double[,][] donneesRepresentees = ImagesPlacees(DonneesARepresenter.ElementAt(i), representation);
                representees.Add(donneesRepresentees);
            }
            return new Tuple<double[,][], List<double[,][]>, double>(carteInitRepresentee, representees, resultOpti.Item2);
        }

        //Integrales
        public static double integrer(double[] x0, double distIntegration, int dimIntegration, double delta, Application f)
        {
            //Integrer la fonction f selon la dim dimIntegration sur une longueur distIntegration en partant de x0, avec un pas de delta
            double somme = 0.0;
            int NbPas = (int)(distIntegration / delta);
            double[] x = copiedb(x0);
            for (int i = 0; i < NbPas; i++)
            {
                somme += delta * f(x);
                x[dimIntegration] += delta;
            }
            return somme;

        }
        public static double integrerPave(double[] x0, double[] xf, double delta, Application f)
        {
            //Integrer la fonction f, sur le pavé entre x0  et xf, avec un pas de delta
            return integrerPave(x0, xf, x0.Length - 1, delta, f);
        }
        private static double integrerPave(double[] x0, double[] xf, int n0, double delta, Application f)
        {
            //Integrer la fonction f de la dim n0 à 0, sur le pavé entre x0  et xf, avec un pas de delta
            if (n0 == 0)
            {
                return integrer(x0, xf[0] - x0[0], 0, delta, f);
            }
            else
            {
                Application integraleF = fparam => integrerPave(fparam, xf, n0 - 1, delta, f);
                return integrer(x0, xf[n0] - x0[n0], n0, delta, integraleF);
            }
        }

        //Normales et pentes
        public static double[] normale(double[] grad)
        {
            //renvoie le vecteur normal (si f va de r^n dans r, la normale est un vecteur de r^n+1) conaissant le gradient
            //le vecteur normal est normal a toutes les derivees partielles du vecteur M(x,y,...,f(x,y,..))
            int Taille = grad.Length;
            double[] retour = new double[Taille + 1];
            retour[Taille] = 1.0;
            for (int i = 0; i < Taille; i++)
            {
                retour[i] = -grad[i];
            }
            double norme = normeVect(retour);
            multiplierVect(ref retour, 1.0 / norme);
            return retour;
        }
        public static double[] normale(double[] x, double delta, Application f)
        {
            //renvoie le vecteur normal (si f va de r^n dans r, la normale est un vecteur de r^n+1)
            double[] grad = gradient(x, delta, f);
            return normale(grad);
        }
        public static double anglePente(double[] normale)
        {
            //Renvoie l'inclinaison en rad de l'hyperplan tangent à la fonction conaissant la normale
            double cos = normale[normale.Length - 1];//produit scalaire avec la derniere dim
            double alpha = Math.Acos(cos);
            return alpha;
        }
        public static double anglePente(double[] x, double delta, Application f)
        {
            //Renvoie l'inclinaison en rad de l'hyperplan tangent à la fonction conaissant la fonction
            return anglePente(normale(gradient(x, delta, f)));
        }

        //Espace
        public static double[] pointImage(double[] x, Application f)
        {
            //retourne le point (x,y,z,...,f(x,y,z,...))
            double[] resultat = new double[x.Length + 1];
            for (int i = 0; i < x.Length; i++)
            {
                resultat[i] = x[i];
            }
            resultat[x.Length] = f(x);
            return resultat;
        }
        public static double[] coordSurHyperplan(double[] M0)
        {
            //Recupere les coordonées du point M0 sur le plan des abscisses (dim n-1), inverse de pointImage
            double[] resultat = new double[M0.Length - 1];
            for (int i = 0; i < M0.Length - 1; i++)
            {
                resultat[i] = M0[i];
            }
            return resultat;
        }
        public static double[] abscisseImpact(double[] M0, double[] direction, double delta, double epsilon, int nb_pas_max, Application f)
        {
            //Retourne l'abscisse du point d'impact avec le graphe de f en partant du point M0, suivant la direction donnée.
            //Avance avec un pas de delta, contact si la distance des points est inferieure a epsilon
            //nb_pas_max, nombre max d'etapes de calcul
            int sens = 1;
            double dist = 0.0;
            double[] ptRecherche;
            for (int i = 0; i < nb_pas_max; i++)
            {
                ptRecherche = sommeVect(M0, droite(direction, sens * dist));
                double dstCritique = Math.Abs(ptRecherche[ptRecherche.Length - 1] - f(coordSurHyperplan(ptRecherche)));
                if (dstCritique <= epsilon)
                {
                    return coordSurHyperplan(ptRecherche);
                }
                sens *= -1;
                if (i % 2 == 0)
                {
                    dist += delta;
                }
            }
            return coordSurHyperplan(M0);
        }

        //Outils optique
        public static double[] vecteurRefraction(double[] abscisse_impact, double[] vect_incident, double n1, double n2, double delta, Application f)
        {
            //Renvoie le vecteur réfracté sur la surface de la fonction image du point abscisse_impact, pour un vecteur incident donné, n1 et n2.
            //delta et f pour les calculs différentiels
            return vecteurRefraction(normale(abscisse_impact, delta, f), vect_incident, n1, n2);
        }
        public static double[] vecteurRefraction(double[] vect_normal, double[] vect_incident, double n1, double n2)
        {
            //renvoie le vecteur réfracté apres impact du vecteur incident sur une surface de vecteur normal donné, n1 et n2
            double cos1 = -1.0 * produitScalaire(vect_incident, vect_normal);
            double k1 = 1.0 - Math.Pow(n1 / n2, 2) * (1.0 - Math.Pow(cos1, 2));
            double k2 = n1 / n2;
            double[] CompIncident = copiedb(vect_incident);
            double[] CompNormale = copiedb(vect_normal);
            if (k1 > 0)
            {
                //Refraction
                double cos2 = Math.Sqrt(k1);
                double k3;
                if (cos1 > 0)
                {
                    k3 = k2 * cos1 - cos2;
                }
                else
                {
                    k3 = k2 * cos1 + cos2;
                }
                multiplierVect(ref CompNormale, k3);
                multiplierVect(ref CompIncident, k2);
                return sommeVect(CompIncident, CompNormale);
            }
            else
            {
                //Reflection totale
                double k4 = 2 * cos1;
                multiplierVect(ref CompNormale, k4);
                return sommeVect(CompNormale, CompIncident);
            }
        }
        public static double[] abscisseImageSuiteRefractions(double[] M0, double[] direction, List<Application> SurfacesRefract, List<double> indicesRefrac, Application Ecran, double delta_diff, double delta_vect, double epsilon, int pas_max)
        {
            //Renvoie l'abscisse du point image apres traversée d'une suite de surfaces de refraction (n), avec la liste des indices de refraction (n+1) et impact sur la surface ecran
            double[] M = copiedb(M0);
            double[] dir = copiedb(direction);
            multiplierVect(ref dir, 1.0 / normeVect(dir));
            double[] absImpact;
            for (int i = 0; i < SurfacesRefract.Count; i++)
            {
                absImpact = abscisseImpact(M, dir, delta_vect, epsilon, pas_max, SurfacesRefract.ElementAt(i));
                dir = vecteurRefraction(absImpact, dir, indicesRefrac.ElementAt(i), indicesRefrac.ElementAt(i + 1), delta_diff, SurfacesRefract.ElementAt(i));
                multiplierVect(ref dir, 1.0 / normeVect(dir));
                M = pointImage(absImpact, SurfacesRefract.ElementAt(i));
            }
            return abscisseImpact(M, dir, delta_vect, epsilon, pas_max, Ecran);
        }
        public static double[] abscisseImageRefraction(double[] M0, double[] direction, Application SurfaceRefract, double n0, double n1, Application Ecran, double delta_diff, double delta_vect, double epsilon, int pas_max)
        {
            //Renvoie l'abscisse du point image apres traversée d'une surface de refraction (n0->n1), et impact sur la surface ecran
            return abscisseImageSuiteRefractions(M0, direction, new List<Application> { SurfaceRefract }, new List<double> { n0, n1 }, Ecran, delta_diff, delta_vect, epsilon, pas_max);
        }
        public static double[] vecteurReflection(double[] abscisse_impact, double[] vect_incident, double n1, double n2, double delta, Application f)
        {
            //Renvoie le vecteur réfléchi sur la surface de la fonction image du point pos_impact, pour un vecteur incident donné, n1 et n2.
            //delta et f pour les calculs différentiels
            return vecteurReflection(normale(abscisse_impact, delta, f), vect_incident, n1, n2);
        }
        public static double[] vecteurReflection(double[] vect_normal, double[] vect_incident, double n1, double n2)
        {
            //renvoie le vecteur réfléchi apres impact du vecteur incident sur une surface de vecteur normal donné, n1 et n2
            double cos1 = -1.0 * produitScalaire(vect_incident, vect_normal);
            double[] CompIncident = copiedb(vect_incident);
            double[] CompNormale = copiedb(vect_normal);
            //Reflection totale
            double k1 = 2 * cos1;
            multiplierVect(ref CompNormale, k1);
            return sommeVect(CompNormale, CompIncident);

        }
        public static NdimArray motifRefraction(double[] direction, List<Application> SurfacesRefract, List<double> indicesRefrac, Application Ecran, double delta_diff, double delta_vect, double epsilon, int pas_max, double[] xmin, double[] xmax, double[] deltasX, double[] ymin, double[] ymax, int[] subdivY)
        {
            //Renvoie le motif (grille de dim n-1) formé sur la surface Ecran apres la traversée des rayons choisi dans le pavé discretisé xmin,xmax,deltasX de dim n, partant avec une direction donnee, de la suite de k surfaces SurfacesRefract et de k+1 indices de refraction
            //Cf absicce image refraction pour delta_diff, delta_vect, epsilon,pas max
            Transformation tr = x0 => abscisseImageSuiteRefractions(x0, direction, SurfacesRefract, indicesRefrac, Ecran, delta_diff, delta_vect, epsilon, pas_max);
            NdimArray impacts = CarteIntensite(xmin, xmax, deltasX, ymin, ymax, subdivY, tr);
            return impacts;
        }

        //Signal
        public static int nexpPow2(int N)
        {
            return (int)Math.Pow(2, Math.Floor(Math.Log(N, 2)) + 1);
        }
        public static double sinc(double x)
        {
            //Sinus cardinal de x
            if(x==0)
            {
                return 1;
            }
            else
            {
                return Math.Sin(x) / x;
            }
        }
        public static double[] DSP(double[] x, int N)
        {
            //Calcul la dsp de x, en N points, entre -fe et fe
            int n = x.GetLength(0);
            double[] retour = new double[N];
            double[] retour_ = new double[N];
            for (int i=0;i<N;i++)
            {
                if(i%100==0)
                Console.WriteLine(i+"/"+N);
                double re = 0;
                double im = 0;
                for(int k=0;k<n;k++)
                {
                    re += x[k] * Math.Cos(-2.0 * Math.PI * (double)k * (double)i / (double)N);
                    im += x[k] * Math.Sin(-2.0 * Math.PI * (double)k * (double)i / (double)N);
                }
                retour[i] = (re * re + im * im)/((double)N);
            }
            for(int i=N/2;i<N;i++)
            {
                retour_[i - N / 2] = retour[i];
                retour_[i] = retour[i-N/2];
            }
            return retour_;
        }
        public static double[] Conv(double[] u, double[] v)
        {
            //Convolue u et v
            int n = u.Length;
            int p = v.Length;
            double[] res = new double[n+p-1];
            for (int i=0;i<n+p+-1;i++)
            {
                double s = 0;
                for(int j=0;j<p;j++)
                {
                    int z = i-p+1+j;
                    if (z < n && z >= 0)
                    {
                        s += v[j] * u[z];
                    }
                }
                res[i] = s;
            }
            return res;
        }
        public static double[] CentrerConv(double[] c, int n)
        {
            //Centre le retour d'un produit de convolution pour qu'il ait une taille n
            double[] retour = new double[n];
            int delta = (c.Length - n) / 2;
            for(int i=0;i<n;i++)
            {
                retour[i] = c[i + delta];
            }
            return retour;
        }
        public static double[] Filtrer(double[] x,int Ordre, double f0, double deltaf, double fe)
        {
            //Filtre le signal x par bande centrée sur f0 et de largeur deltaf, avec un ordre donné, sachant la frequence d'echantillonage fe 
            double[] filtre = new double[Ordre];
            for(int i=0;i<Ordre;i++)
            {
                double t = ((double)i - (Ordre / 2.0)) / fe;
                double cos = Math.Cos(2.0 * Math.PI * f0 * t);
                double sincard = sinc(Math.PI * deltaf * t)* (deltaf / fe);
                filtre[i] = cos*sincard;
            }
            return CentrerConv(Conv(x, filtre),x.Length);
        }
        public static double[] TransposerFreq(double[] x, double fp,double fe)
        {
            //Transpose le signal x sur fp et -fp, sachant fe la frequence d'echantillonage
            double[] ret = copiedb(x);
            for(int i=0;i<ret.Length;i++)
            {
                double t = ((double)i - (ret.Length / 2.0)) / fe;
                double cos = Math.Cos(2.0 * Math.PI * fp * t); ;
                ret[i] = ret[i] * cos;
            }
            return ret
;
        }
        public static double[] Transmettre(double[,] messages,int ordreFiltre, double fe)
        {
            //Transforme un ensemble de signaux (lignes) en un seul signal temporel modulé en amplitude, sachant la frequence d'echantillonage fe
            int NbMessages = messages.GetLength(0);
            int LongMessages = messages.GetLength(1);
            double deltaf = fe / (2.0 * NbMessages);
            double[] temp = Vect1Val(0,LongMessages);
            for(int i=0;i<NbMessages;i++)
            {
                double fp = (i+0.5)*deltaf;
                temp = sommeVect(temp, TransposerFreq(Filtrer(ligne(messages,i),ordreFiltre,0,deltaf,fe),fp,fe));
            }
            return temp;
        }
        public static double[,] Recevoir(double[] signal, int NbMessages, int ordreFiltre, double fe)
        {
            //Inverse de la fonction Transmettre, signal non lissé
            return Recevoir(signal, NbMessages, ordreFiltre, fe, true, 0);
        }
        public static double[,] Recevoir(double[] signal, int NbMessages,int ordreFiltre, double fe,bool Analogique, double Ts)
        {
            //Inverse de la fonction Transmettre, si non Analogique, le signal est lissé entre {-1, 1} 
            int LongMessages = signal.GetLength(0);
            int p = LongMessages;
            double[,] retour = new double[NbMessages, p];
            double deltaf = fe / (2.0 * NbMessages);
            for (int i = 0; i < NbMessages; i++)
            {
                double fp = (i + 0.5) * deltaf;
                double[] ligne = Filtrer(signal, ordreFiltre, fp, deltaf, fe);
                ligne = TransposerFreq(ligne, fp, fe);
                ligne = Filtrer(ligne, ordreFiltre, 0, deltaf, fe);
                if (!Analogique)
                {
                    ligne = BinVersAnalog(AnalogVersBin(ligne,Ts,fe), Ts, fe);
                }
                for(int q=0;q<p;q++)
                {
                    retour[i, q] = ligne[q];
                }
            }
            return retour;
        }
        public static double[] BinVersAnalog(bool[] bits,double Ts, double fe)
        {
            //Convertit un tableau de bits en signal analogique de periode TS, pour une frequence d'echantillonage fe
            int Ns = (int)(Ts * fe);
            int n = Ns * bits.Length;
            double[] retour = new double[n];
            int k = 0;
            for(int i=0;i<bits.Length;i++)
            {
                double val;
                if(bits[i])
                {
                    val = 1;
                }
                else
                {
                    val = -1;
                }
                for (int j = 0; j < Ns; j++)
                {
                    retour[k] = val;
                    k++;
                }
            }
            return retour;
        }
        public static bool[] AnalogVersBin(double[] sig, double Ts, double fe)
        {
            //Convertit un signal analogique de periode TS en tableau de bool, pour une frequence d'echantillonage fe
            int Ns = (int)(Ts * fe);
            int n =sig.Length/Ns;
            bool[] retour = new bool[n];
            int k = 0;
            for (int i = 0; i < n; i++)
            {
                double val=0;
                for (int j = 0; j < Ns; j++)
                {
                    val += sig[k];
                    k++;
                }
                retour[i] = val>0;
            }
            return retour;
        }
        public static double PuissanceSignal(double[] sig, double fe)
        {
            //Calcule la puissance su signal sig, fe = frequence echantillonage
            //Parseval
            return Math.Pow(normeVect(sig), 2) / sig.Length;
        }
        public static double[] BruiterSignal(double[] sig,double RSB, double fe, ref Random r)
        {
            //Bruite le signal avec un bruit additif gaussien pour un certain rsb
            double ps = PuissanceSignal(sig, fe);
            double pb = ps/Math.Pow(10, RSB / 10.0);
            double sigma = Math.Sqrt(pb);
            double[] result = new double[sig.Length];
            for(int i=0;i<sig.Length;i++)
            {
                double r1 = Math.Max(r.NextDouble(),double.Epsilon);
                double r2 = r.NextDouble();
                double rg = Math.Sqrt(-2.0*Math.Log(r1))*Math.Sin(2.0*Math.PI*r2)*sigma;
                result[i] = sig[i] + rg;
            }
            return result;
        }
        
        //fonctions vecteurs
        public static double[] projeter(double[] u, double[] v)
        {
            //Projette V sur U
            double[] res = copiedb(u);
            multiplierVect(ref res, produitScalaire(u, v) / produitScalaire(u, u));
            return res;
        }
        public static double[] normerVect(double[] u)
        {
            double[] un = copiedb(u);
            multiplierVect(ref un, 1.0 / normeVect(un));
            return un;
        }
        private static double produitScalaire(double[] V1, double[] V2)
        {
            //Produit scalaire de deux vecteurs
            int Taille = Math.Min(V1.Length, V2.Length);
            double Somme = 0.0;
            for (int i = 0; i < Taille; i++)
            {
                Somme += V1[i] * V2[i];
            }
            return Somme;
        }
        private static double normeVect(double[] V)
        {
            //Norme du vecteur v
            return Math.Sqrt(produitScalaire(V, V));
        }
        private static void multiplierVect(ref double[] V, double k)
        {
            //Multiplier le vecteur par k
            int Taille = V.Length;
            for (int i = 0; i < Taille; i++)
            {
                V[i] *= k;
            }
        }
        private static double[] copiedb(double[] V1)
        {
            //renvoie une copie de V1
            int Taille = V1.Length;
            double[] Resultat = new double[Taille];
            for (int i = 0; i < Taille; i++)
            {
                Resultat[i] = V1[i];
            }
            return Resultat;
        }
        private static double[] sommeVect(double[] V1, double[] V2)
        {
            //renvoie une somme de V1 et V2
            int Taille = Math.Min(V1.Length, V2.Length);
            double[] Resultat = new double[Taille];
            for (int i = 0; i < Taille; i++)
            {
                Resultat[i] = V1[i] + V2[i];
            }
            return Resultat;
        }
        private static double distanceVect(double[] V1, double[] V2)
        {
            //renvoie la distance entre V1 et V2
            int Taille = Math.Min(V1.Length, V2.Length);
            double Resultat = 0;
            for (int i = 0; i < Taille; i++)
            {
                Resultat += Math.Pow(V1[i] - V2[i], 2);
            }
            Resultat = Math.Sqrt(Resultat);
            return Resultat;
        }
        private static double[] droite(double[] direction, double dist)
        {
            //renvoie un vecteur de longueur dist en direction de direction
            int Taille = direction.Length;
            double nrme = normeVect(direction);
            double[] Resultat = new double[Taille];
            for (int i = 0; i < Taille; i++)
            {
                Resultat[i] = direction[i] * (dist / nrme);
            }
            return Resultat;
        }
        private static double[] vectAlea(ref Random r, double[] min, double[] max)
        {
            double[] retour = new double[min.Length];
            for (int i = 0; i < min.Length; i++)
            {
                retour[i] = min[i] + r.NextDouble() * (max[i] - min[i]);
            }
            return retour;
        }

        //Tableau custom
        public class NdimArray
        {
            //Tableau déclaré avec un nombre de dimentions variables
            private int nbDim;
            private int[] taille;
            private double[] elements;

            public NdimArray(int[] taille)
            {
                this.taille = taille;
                nbDim = taille.Length;
                int p = 1;
                for (int i = 0; i < nbDim; i++)
                {
                    p *= taille[i];
                }
                elements = new double[p];
            }

            public void zeros()
            {
                for (int i = 0; i < elements.Length; i++)
                {
                    elements[i] = 0.0;
                }
            }
            private int get_rang(int[] position)
            {
                int n = 0;
                int p = 1;
                for (int i = 0; i < nbDim; i++)
                {
                    if (position[i] >= taille[i] || position[i] < 0)
                    {
                        return -1;
                    }
                    n += p * position[i];
                    p *= taille[i];
                }
                return n;
            }
            public double get(int[] position)
            {
                int pos = get_rang(position);
                if (pos == -1)
                {
                    return 0.0;
                }
                else
                {
                    return elements[pos];
                }
            }
            public bool set(int[] position, double val)
            {
                int pos = get_rang(position);
                if (pos != -1)
                {
                    elements[pos] = val;
                    return true;
                }
                else
                {
                    return false;
                }
            }
            public bool add(int[] position, double val)
            {
                int pos = get_rang(position);
                if (pos != -1)
                {
                    elements[pos] += val;
                    return true;
                }
                else
                {
                    return false;
                }
            }
            public void normaliser()
            {
                double max = 0;
                for (int i = 0; i < elements.Length; i++)
                {
                    if (Math.Abs(elements[i]) > max)
                    {
                        max = Math.Abs(elements[i]);
                    }
                }
                if (max != 0)
                {
                    for (int i = 0; i < elements.Length; i++)
                    {
                        elements[i] /= max;
                    }
                }
            }
            public int get_NbDim()
            {
                return nbDim;
            }
            public int[] get_Taille()
            {
                return taille;
            }
        }
        //Infos pour optimisation
        public struct InfoOpti
        {
            public double[] param_min;
            public double[] param_max;
            public int nbStagnationsMax;
            public int nbDepartsMax;
            public int iterMax_newton;
            public double relStop_newton;
            public double delta_diff;
            public TypeOpti typeOpti;
            public double pasGrad;

            public InfoOpti(double[] param_min, double[] param_max, int nbStagnationsMax, int nbDepartsMax, int iterMax_newton, double relStop_newton, double delta_diff, TypeOpti typeOpti, double pasGrad)
            {
                this.param_min = param_min;
                this.param_max = param_max;
                this.nbStagnationsMax = nbStagnationsMax;
                this.nbDepartsMax = nbDepartsMax;
                this.iterMax_newton = iterMax_newton;
                this.relStop_newton = relStop_newton;
                this.delta_diff = delta_diff;
                this.typeOpti = typeOpti;
                this.pasGrad = pasGrad;
            }
            public InfoOpti(double[] param_min, double[] param_max, int nbStagnationsMax, int nbDepartsMax, int iterMax_newton, double relStop_newton, double delta_diff, TypeOpti typeOpti)
            {
                this.param_min = param_min;
                this.param_max = param_max;
                this.nbStagnationsMax = nbStagnationsMax;
                this.nbDepartsMax = nbDepartsMax;
                this.iterMax_newton = iterMax_newton;
                this.relStop_newton = relStop_newton;
                this.delta_diff = delta_diff;
                this.typeOpti = typeOpti;
                this.pasGrad = delta_diff;
            }
        }
    }
}
