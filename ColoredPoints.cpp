// -- Main Libraries
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <ctime> 

// -- Parallelism libraries
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// -- Read DataSet libraries
#include <sstream>
#include <string>
#include <fstream>

// -- VTK
#include <vtkActor.h>
#include <vtkCellArray.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkUnsignedCharArray.h>
#include <vtkVertexGlyphFilter.h>

#include <vtkNamedColors.h>

// For compatibility with new VTK generic data arrays
#ifdef vtkGenericDataArray_h
#define InsertNextTupleValue InsertNextTypedTuple
#endif

using namespace std;

template<int ndim>
struct Point{
    float cords[ndim];

    Point(){
        for(int i = 0; i < ndim; i++){
            cords[i] = 0;
        }
    }

    Point( float _cords[] ){
        for (int i = 0; i < ndim; i++){
            cords[i] = _cords[i];
        }
    }

    float &operator[] (int posicion){
        return cords[posicion];
    }
};

template<int ndim>
struct MBR{
    Point<ndim> bottomLeft;
    Point<ndim> upperRight;

    MBR(){
        bottomLeft = Point<ndim>();
        upperRight = Point<ndim>();
    }
};

template<int ndim>
struct ClusterGroup{
    //############### VARIABLES ###################
    Point<ndim> centroid; // Punto / centroide
    vector<Point<ndim>> dataCluster; // Puntos de UN cluster
    vector<Point<ndim>> dataClusterAux; // Para la verificacion de cambios

    //############### CONSTRUCTOR ###################
    ClusterGroup(){
        centroid = Point<ndim>();
    }

    //############### CLUSTER ###################
    void createCentroid(Point<ndim> bottomLeft, Point<ndim> upperRight){
        // --> Create Centroid
        for (int j = 0; j < ndim; j++) {// Random number per axis
            float minRange = bottomLeft[j];
            float maxRange = upperRight[j];

            float randomNumber = minRange + static_cast<float>(rand()) * static_cast<float>(maxRange-minRange) / RAND_MAX;
            centroid[j] = randomNumber;
        }
    }

    //############### PRINT ###################
    void printCentroid(){
        for(int i = 0; i < ndim; i++){
            cout << centroid[i] << " - ";
        }
        cout << endl;
    }

    void printDataMAIN(){
        for (int i = 0; i < dataCluster.size(); i++){
            cout << "Dato " << i+1 << ": ";
            for(int j = 0; j < ndim; j++){
                cout << dataCluster[i][j] << " - ";
            }
            cout << endl;
        }
    }

    void printDataAUX(){
        for (int i = 0; i < dataClusterAux.size(); i++){
            cout << "Dato " << i+1 << ": ";
            for(int j = 0; j < ndim; j++){
                cout << dataClusterAux[i][j] << " - ";
            }
            cout << endl;
        }
    }
};

template<int ndim>
struct Kmeans{
    //############### VARIABLES ###################
    int kCentroides; // numero de clusters

    vector<Point<ndim>> dataSet; // Puntos del DATASET
    MBR<ndim> minimumRectangle; // Minimio Rectangulo

    vector<ClusterGroup<ndim>> groupClusters; // Clusters

    //############### For VTK ###################
    vector<int> numberDataVTK;
    vector<Point<ndim>> dataSetOrderedVTK;

    //############### CONSTRUCTOR ###################
    Kmeans( int _kCentroides ){
        kCentroides = _kCentroides;
    }

    //############### PRINT ###################
    void printPoint( Point<ndim>  pointAux ){
        for(int i = 0; i < ndim; i++){
            cout << pointAux[i] << " - ";
        }
        cout << endl;
    }

    void printAllData(){
        for(int i = 0; i < dataSet.size(); i++){
            cout << i << ". -> ";
            printPoint(dataSet[i]);
        }
    }

    void printMBR(){
        cout << "BottomLeft...";
        for(int i = 0; i < ndim ; i++){
            cout << minimumRectangle.bottomLeft[i] << " - ";
        }
        cout << endl << "UpperRight...";
        for(int i = 0; i < ndim; i++){
            cout << minimumRectangle.upperRight[i] << " - ";
        }
    }

    void printClusteredData(){
        for(int i = 0; i < groupClusters.size(); i++){
            cout << "Data of Cluster " << i+1 << ": " << endl;
            groupClusters[i].printDataMAIN();
            cout << endl;
        }
    }

    //############### FUNCIONES ADICIONALES ###################
    void calculareMBR(){
        Point<ndim> bottomLeft = Point<ndim>();
        Point<ndim> upperRight = Point<ndim>();

        for(int i = 0; i < ndim; i++){ // for all axis
            float maxData = dataSet[i][0];
            float minData = dataSet[i][0];
            for(int j = 1; j < dataSet.size(); j++){ // compare with all data
                float maxDataAux = dataSet[j][i];
                float minDataAux = dataSet[j][i];

                // Minimum...
                if( minDataAux < minData ){
                    minData = minDataAux;
                }

                // Maximum
                if(maxDataAux > maxData){
                    maxData = maxDataAux;
                }
            }

            // Put Results in variables
            bottomLeft[i] = minData;
            upperRight[i] = maxData;
        }

        minimumRectangle.bottomLeft = bottomLeft;
        minimumRectangle.upperRight = upperRight;
    }

    float getEuclideanDistance(Point<ndim> dato, Point<ndim> centroid){
        float res = 0;
        for(int i = 0; i < ndim; i++){
            res = res + pow( centroid[i]-dato[i] , 2);
        }
        return sqrt(res);
    }

    bool compareTwoPoints( Point<ndim> p1, Point<ndim> p2 ){
        // Return true if they are the same
        // Return false if they are different
        for(int i = 0; i < ndim; i++){
            if(p1[i] != p2[i]){
                return false;
            }
        }
        return true;
    }

    void insertPoint(Point<ndim> pointA){
        dataSet.push_back(pointA);
    }

    //############### KMEANS ###################
    void labelData(){
        // 1. Borrar los datos de AUX
        for(int i = 0; i < groupClusters.size(); i++){
            int auxDelete = groupClusters[i].dataClusterAux.size();
            for(int j = 0; j < auxDelete; j++){
                groupClusters[i].dataClusterAux.pop_back();
            }
        }

        // 2. Copiar los puntos del MAIN al AUX
        for(int i = 0; i < groupClusters.size(); i++){
            groupClusters[i].dataClusterAux.insert(groupClusters[i].dataClusterAux.begin(), groupClusters[i].dataCluster.begin(), groupClusters[i].dataCluster.end());
        }

        // 3. Borrar los datos del MAIN
        for(int i = 0; i < groupClusters.size(); i++){
            int auxDelete = groupClusters[i].dataCluster.size();
            for(int j = 0; j < auxDelete; j++){
                groupClusters[i].dataCluster.pop_back();
            }
        }

        // 4. Re-Etiquetar los puntos
        for(int i = 0; i < dataSet.size(); i++){
            int posMinClusterDistance = 0;
            float distance = getEuclideanDistance(dataSet[i], groupClusters[0].centroid);
            for(int j = 1; j < groupClusters.size(); j++){
                float distanceAux = getEuclideanDistance(dataSet[i], groupClusters[j].centroid);
                if(distanceAux < distance){
                    distance = distanceAux;
                    posMinClusterDistance = j;
                }
            }
            groupClusters[posMinClusterDistance].dataCluster.push_back(dataSet[i]);
        }
    }

    void createClusters(){
        // 1. Crear los centroides guardarlos en un vector
        srand(static_cast<unsigned> (time(NULL)));
        for(int i = 0; i < kCentroides; i++){
            ClusterGroup<ndim> clusterAux;
            clusterAux.createCentroid(minimumRectangle.bottomLeft, minimumRectangle.upperRight);

            //cout << endl << "Centroid of Cluster: " << i+1 << endl;
            //clusterAux.printCentroid();

            groupClusters.push_back(clusterAux);
        }

        // 2. Re-Etiquetar los puntos
        labelData();
    }

    void moveCentroids(){
        for(int i = 0; i < groupClusters.size(); i++){ // Recorrer todos los clusters
            for( int j = 0; j < ndim; j++){ // Recorer las ndimensiones
                float suma = 0;
                float promedio = 0;
                for(int k = 0; k < groupClusters[i].dataCluster.size(); k++){ // Recorrer todos los datos de un cluster
                    suma = suma + groupClusters[i].dataCluster[k][j];
                }
                promedio = suma/groupClusters[i].dataCluster.size();
                groupClusters[i].centroid[j] = promedio;
            }
        }

        /*
        // Print
        for(int i = 0; i < kCentroides; i++) {
            cout << endl << "New Centroid of Cluster: " << i + 1 << endl;
            groupClusters[i].printCentroid();
        }
        cout << endl;
         */
    }

    bool checkChanges(){
        // Return true if something changes
        // Return false if nothing changes
        bool flag;
        for(int i = 0; i < groupClusters.size(); i++){
            // 1. Verificar si tienen la misma cantidad de puntos
            if( groupClusters[i].dataCluster.size() != groupClusters[i].dataClusterAux.size() ){
                return true; // Es diferente tamaño!!!!
            }

            // 2. Verificar punto a punto si cambio
            else{
                for(int j = 0; j < groupClusters[i].dataCluster.size(); j++){
                    flag = compareTwoPoints(groupClusters[i].dataCluster[j], groupClusters[i].dataClusterAux[j]);
                    if(flag == false){ // Si es que los puntos son diferentes
                        return true; // Diferentes puntos!!!!
                    }
                }
            }
        }
        return false; // No hay ningun cambio. STOP
    }

    void constructor(){
        cout << endl <<  "---------- Creando Estructura ------------" << endl;

        // 1. Obteniendo limites
        cout << "1. MBR:" << endl;
        calculareMBR();
        printMBR();

        // 2. Create cluster
        cout << endl << endl << "2. Create Clusters " << endl;
        createClusters();
        //printClusteredData();

        // --> Bucle Principal
        bool flag;
        int contador = 0;
        cout << endl << endl << "3. Bucle principal" << endl;
        do {
            contador++;
            // 3. Mover centroides
            moveCentroids();

            // 4. Label Data
            labelData();

            // --> Print Result
            //printClusteredData();

            // --> Check changes
            flag = checkChanges();

        } while ( flag );

        cout << endl << "El numero de veces que se hizo el loop es: " << contador << endl;
    }

    void drawPoints(){
        for(int i = 0; i < groupClusters.size(); i++){ // Recorrer todos los clusters creados
            int numberPoints = groupClusters[i].dataCluster.size();
            for(int j = 0; j < numberPoints; j++){
                dataSetOrderedVTK.push_back(groupClusters[i].dataCluster[j]);
            }
            numberDataVTK.push_back(numberPoints);
        }
    }
};

int main(int, char*[])
{
    // #################### MAIN ##############################
    int kCentroides = 2;
    const int ndim = 2; // VALOR FIJO
    Kmeans<ndim> test1(kCentroides);

    cout << endl << "######################################## " << endl;
    cout << "               K - Means " << endl;
    cout << "######################################## " << endl;

    //string filename = "dataSet_Test.txt";
    string filename = "puntos_2_bloques.txt";
    //string filename = "puntos_5_bloques.txt";
    ifstream in(filename);
    float dim_x, dim_y;

    unsigned t0, t1;
    t0 = clock();

    while( in >> dim_x >> dim_y){
        float pointAux[] = {dim_x, dim_y};
        Point<ndim> PntoAux(pointAux);

        test1.insertPoint(pointAux);
    }

    //cout << "Data: " << endl;
    //test1.printAllData();

    test1.constructor();

    t1 = clock();    
    double time = (double(t1-t0)/CLOCKS_PER_SEC);
    cout << "Execution Time: " << time << endl;

    cout << "Fin del Programa" << endl;

    test1.drawPoints();

    // #################### VTK ##############################
    vtkNew<vtkPoints> points;
    for(int i = 0; i < test1.dataSetOrderedVTK.size(); i++){
        points->InsertNextPoint(test1.dataSetOrderedVTK[i][0], test1.dataSetOrderedVTK[i][1], 0.0);
    }

    vtkNew<vtkPolyData> pointsPolydata;

    pointsPolydata->SetPoints(points);

    vtkNew<vtkVertexGlyphFilter> vertexFilter;
    vertexFilter->SetInputData(pointsPolydata);
    vertexFilter->Update();

    vtkNew<vtkPolyData> polydata;
    polydata->ShallowCopy(vertexFilter->GetOutput());

    // Setup colors
    vtkNew<vtkNamedColors> namedColors;

    vtkNew<vtkUnsignedCharArray> colors;
    colors->SetNumberOfComponents(3);
    colors->SetName("Colors");

    string colores[] = {"Tomato", "Black", "Blue", "Orange", "Green"};
    int contador = 0;
    for(int i = 0; i < kCentroides; i++){
        int limite = test1.numberDataVTK[contador];
        for(int j = 0; j < limite; j++){
            colors->InsertNextTupleValue(namedColors->GetColor3ub(colores[contador]).GetData());
        }
        contador++;
    }
    polydata->GetPointData()->SetScalars(colors);

    // Visualization
    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputData(polydata);

    vtkNew<vtkActor> actor;
    actor->SetMapper(mapper);
    actor->GetProperty()->SetPointSize(10);

    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderWindow> renderWindow;
    renderWindow->AddRenderer(renderer);
    renderWindow->SetWindowName("ColoredPoints");

    vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
    renderWindowInteractor->SetRenderWindow(renderWindow);

    renderer->AddActor(actor);
    renderer->SetBackground(namedColors->GetColor3d("White").GetData());
    renderWindow->Render();
    renderWindowInteractor->Start();

    return EXIT_SUCCESS;
}
