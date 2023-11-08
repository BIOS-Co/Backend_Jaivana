#Fast API:
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
#packages:
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
from datetime import date


#Packages ML Model:
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import joblib

app = FastAPI()

#CORDS:

"""
CORDS
"""
origins = [
    "http://localhost:3000",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#FUNCTIONS:

class RecomendarProductosRequest(BaseModel):
    nit_del_cliente: str
    departamento: str = None
    ciiu: int = None
    seccion: str = None
    producto_1: str = None
    producto_2: str = None
    producto_3: str = None
    producto_4: str = None
    producto_5: str = None
    producto_6: str = None
    producto_7: str = None
    producto_8: str = None
    producto_9: str = None
    producto_10: str = None
    producto_11: str = None
    producto_12: str = None
    producto_13: str = None
    producto_14: str = None
    producto_15: str = None

def combinations(list1,list2):
  """
  DESCRIPTION: Function that find all the possible combinations between two lists.
  -----------------------------------------------------------------------------------
  -----------------------------------------------------------------------------------
  PARAMETERS:
  - list1(List):List 1
  - list2(List): List 2
  -----------------------------------------------------------------------------------
  -----------------------------------------------------------------------------------
  RETURN:
  - possible_combinations (Dict): Dictionary in the wich the keys are all the possibles combinations between the two lists


  """
  possible_combinations={}
  for i in list1:
    for j in list2:
      possible_combinations[i+"_"+j]=[]

  return possible_combinations


def article_similarity_matrix(data_ventas):
  """
  DESCRIPTION: Function that serves for find the article similarity matrix of a DataFrame of sales
  -------------------------------------------------------------------------------------------------
  -------------------------------------------------------------------------------------------------
  PARAMETERS:
  - data_ventas (DataFrame): DataFrame of sales.
  -------------------------------------------------------------------------------------------------
  -------------------------------------------------------------------------------------------------
  RETURN:
  item_item_sim_matrix (DataFrame): article similarity matrix

  """

  #Matrix Customer Vs Product:
  customer_item_matrix = data_ventas.pivot_table(
      index='nit',
      columns='codigo',
      values='cantidad',
      aggfunc='sum')
  customer_item_matrix = customer_item_matrix.applymap(lambda x: 1 if x > 0 else 0) #La celda tomará el valor de 1, si el cliente alguna vez compró el producto (sin importar la cantidad) y tomará el valor de 0 sie l cliente nunca compró el producto.


  item_item_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix.T)) #similarity matrix between articles

  item_item_sim_matrix.columns = customer_item_matrix.T.index #change the name of the columns for the stockcode
  #change the name of the rows for the stockcode:
  item_item_sim_matrix['codigo'] = customer_item_matrix.T.index
  item_item_sim_matrix = item_item_sim_matrix.set_index('codigo')

  return item_item_sim_matrix

def tipificasion(variable_1,variable_2,data_frame):
  """
  DESCRIPTION: This function serves for find the Dataframes that is in every combination.
  ---------------------------------------------------------------------------------------
  ---------------------------------------------------------------------------------------
  PARAMETERS:
  - variable_1 (str): Variable1 for do the tipificasion.
  - variable_1 (str): Variable2 for do the tipificasion.
  - data_frame (DataFrame): DataFrame with all data.
  ---------------------------------------------------------------------------------------
  ---------------------------------------------------------------------------------------
  RETURN:
  possible_combinations (Dict): Dictionary in the which the keys are every combination and the values are the Dataframe with the instances that have that combination.


  """
  possible_combinations=combinations(data_frame[variable_1].unique(),data_frame[variable_2].unique())

  for i in data_frame.index:
    combination=data_frame[variable_1].loc[i]+"_"+data_frame[variable_2].loc[i]
    possible_combinations[combination].append(i)

  for i in possible_combinations.keys():
    possible_combinations[i]=[data_frame.loc[possible_combinations[i]]]
    try:
      possible_combinations[i].append(article_similarity_matrix(possible_combinations[i][0]))
    except:
      possible_combinations[i]


  return possible_combinations



#FUNCTIONS:
def recomend_products_for_article(combination,code_article,possible_combinations):
  """
  DESCRIPTION:This function serves for recommend the complementary products of a product, this recommend the products that have a simmilarity metric
  majour a 0.7 with the product bought. If the number of recommended products is less than 5, we recommend the 5 products more similar without care the similitu metric.
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  PARAMETERS:
  - combination (str):String with the combination ubication_EconomicActivity.
  - code_article (int): Code of the article that the customer bought.
  - possible_combinations (Dict): Dictionary in the which the keys are every combination and the values are the Dataframe with the instances that have that combination and the respective matrix similarity between products
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  RETURN:
  - recom (List): List with the codes of the articles for recommend.

  """
  recom=[]
  for i in possible_combinations[combination][1][code_article].index:
    if possible_combinations[combination][1][code_article].loc[i]>=0.7:
      recom.append(i)
  if len(recom)<5:
    recom=[]
    count=0
    for i in possible_combinations[combination][1].sort_values(code_article,ascending=False)[code_article].index:
      if count<5:
        recom.append(i)
  return recom

def recomend_codes_for_csv(ventas,ubicacion,A_economica,possible_combinations):
  """
  DESCRIPTION:This function serves for recommend the complementary products of varios products, this recommend the products that have a simmilarity metric
  majour a 0.7 with the product bought for evey product. If the number of recommended products is less than 5, we recommend the 5 products more similar without care the similitu metric for every product.
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  PARAMETERS:
  - ventas (DataFrame): DataFrame with the information of the bought.
  - ubicacion (str): Ubicacion of the customer.
  - A_economica (str): Economic activity of the customer.
  - possible_combinations (Dict): Dictionary in the which the keys are every combination and the values are the Dataframe with the instances that have that combination and the respective matrix similarity between products
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  RETURN:
  - recomendacion (List): List with the codes of the articles for recommend.

  """

  recomendacion=[]
  for i in ventas["ARTICULOS_COMPRADOS"].unique():
    combination=(ubicacion+"_"+str(A_economica)).upper()
    combination=combination.replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace('Ú','U')
    try:
      recom=recomend_products_for_article(combination,i,possible_combinations)
      recomendacion=recomendacion+recom
    except KeyError:
      recomendacion=recomendacion
  recomendacion=list(set(recomendacion))
  return recomendacion

def recomend_name_products_for_csv(data_ventas,recomendacion):
  """
  DESCRIPTION:This function serves for get a DataFrame with the codes and the respective names of the recomemended products.
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  PARAMETERS:
  - data_ventas (DataFrame): DataFrame with the history of sales of the company.
  - recomendacion (List): List with the codes of the products for recommend.
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  RETURN:
  - articulos_recomendados (DataFrame): DataFrame with the codes and the respective names of the recomemended products.

  """
  name_articulos=[]
  code_articulos=[]
  for i in recomendacion:
    articulo=data_ventas[data_ventas["codigo"]==i]["n_articulo"].unique()[0]
    name_articulos.append(articulo)
    code_articulos.append(i)

  articulos_recomendados=pd.DataFrame({"codigo":code_articulos,"nombre articulo":name_articulos})
  return articulos_recomendados

def generate_recomendation(possible_combinations,data_ventas,type_tipificasion,Departamento,Ciiu,ventas,region,seccion):
  """
  DESCRIPTION:This function serves for get a DataFrame and a csv file with the codes and the respective names of the recomemended products.
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  PARAMETERS:
  - path_csv (str): Path of the csv file with the information of the bought.
  - possible_combinations (Dict): Dictionary in the which the keys are every combination and the values are the Dataframe with the instances that have that combination and the respective matrix similarity between products
  - data_ventas (DataFrame): DataFrame with the history of sales of the company.
  - type_tipificasion (str): Stirng for indicate the type of tipificasion.
  - count (int): This parameter serves for indicate if the function should o shouldn´t ask for the Departamento and the Ciiu.
  - Deparatamento (str): Departamento of the customer.
  - Ciiu (str): Ciiu of the customer.
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  RETURN:
  - articulos_recomendados (DataFrame): DataFrame with the codes and the respective names of the recomemended products.
  - Deparatamento (str): Departamento of the customer.
  - Ciiu (str): Ciiu of the customer.

  """


  if type_tipificasion=="DEPARTAMENTO_CIIU":
    ubicacion=Departamento
    A_economica=Ciiu
  elif type_tipificasion=="REGION_CIIU":
    ubicacion=region
    A_economica=Ciiu
  elif type_tipificasion=="DEPARTAMENTO_SECCION":
    ubicacion=Departamento
    A_economica=seccion

  elif type_tipificasion=="REGION_SECCION":
    ubicacion=region
    A_economica=seccion


  codes_recomend=recomend_codes_for_csv(ventas,ubicacion,A_economica,possible_combinations)
  articulos_recomendados=recomend_name_products_for_csv(data_ventas,codes_recomend)
  articulos_recomendados.to_csv("data/Articulos_recomedados.csv",sep=";",index=False)
  return articulos_recomendados


#PREPEDIDO FUNCTION:
def preproduct_recommend(path_compras,data_ventas_with_dates):
  """
  DESCRIPTION: This function serves for get a DataFrame and a csv file with the prepedido for a customer.
  ---------------------------------------------------------------------------------------------------------
  ---------------------------------------------------------------------------------------------------------
  PARAMETERS:
  - path_compras (str): Path of the csv file with the information of the customer.
  - data_ventas_with_dates (DataFrame): DataFrame with the historical sales of the company (including dates)
  ---------------------------------------------------------------------------------------------------------
  ---------------------------------------------------------------------------------------------------------
  RETURN:
  - prepedido (DataFrame): DataFrame with the prepedido. If the customer doesn´t have historical for the last year, the DataFrame and the csv file contents:
  El cliente no tiene historial de compras durante el último año; por lo cual no es posible hacer una recomendación de preproducto.


  """
  try:
    compras=pd.read_csv(path_compras,sep=";")
    compras.columns=["ARTICULOS_COMPRADOS","NIT"]
  except:
    compras=pd.read_csv(path_compras,sep=",")
    compras.columns=["ARTICULOS_COMPRADOS","NIT"]

  nit_customer=int(compras["NIT"].loc[0])


  # Become the columns "fecha" to format date:
  data_ventas_with_dates['fecha'] = pd.to_datetime(data_ventas_with_dates['fecha'], format='%Y-%m-%d')

  # Date Today:
  date_today=date.today()

  #Date today one year ago:
  one_year_ago = date_today.replace(year=date_today.year - 1) # fecha de hace un año
  one_year_ago = pd.Timestamp(str(one_year_ago))


  # Filtering only the sales of the last year:
  mask = data_ventas_with_dates['fecha'] > one_year_ago
  data_ventas_last_year_with_dates=data_ventas_with_dates[mask]


  #Filtering only the sales of the last year of a customer:
  data_ventas_customer_last_year=data_ventas_last_year_with_dates[data_ventas_last_year_with_dates["nit"]==nit_customer]

  #Dictionary in the which the keys are the products that the customer has bought and the values is the amount that bought of the respective product:
  productos={}
  for i in data_ventas_customer_last_year["codigo"].unique():
    ventas=data_ventas_customer_last_year[data_ventas_customer_last_year["codigo"]==i]
    cantidad=ventas["cantidad"].sum()
    productos[i]=cantidad

  #Median:
  median=np.median(np.array(list(productos.values())))

  #Se recomiendan productos (cuya cantidad de compra) están por encima de la mediana de las cantidades de compras (de todos los productos):
  code_products_for_recomend=[]
  products_for_recomend=[]
  for i in productos.keys():
    if productos[i]>=median:
      code_products_for_recomend.append(i)
      products_for_recomend.append(str(data_ventas_customer_last_year[data_ventas_customer_last_year["codigo"]==i]["n_articulo"].unique()[0]))
  prepedido=pd.DataFrame({"Codigo Prepedido":code_products_for_recomend,"nombre articulo Prepedido":products_for_recomend})
  if prepedido.shape[0]==0:
    prepedido=pd.DataFrame({"Observacion":["El cliente no tiene historial de compras durante el último año; por lo cual no es posible hacer una recomendación de preproducto."]})
  prepedido.to_csv("data/prepedido.csv",sep=";",index=False)
  return prepedido

#ML Model Functions:
def sort_array_desc(arr):
    # Create a copy of the array to avoid modifying the original array
    arr_copy = arr.copy()
    # Get the indices that would sort the array in descending order
    sorted_indices = np.argsort(arr_copy)[::-1]
    return sorted_indices


def get_recommendation(product_code, departamento, sector, subsector,pipeline,models):

  X = np.array([departamento, sector, subsector]).reshape(1,-1)

  result = sort_array_desc(pipeline.transform(X))[0]


  for group in result:

    matrix = models[group]
    sim = matrix.loc[product_code].sort_values(ascending=False)
    sim = sim.drop(product_code)
    if sim[sim>0].shape[0] != 0:
      if sim[sim>0.7].shape[0] >= 5:
        return sim[sim>0.7].index.to_list()
      else:
        return sim[:5]
  return []


def generate_recomendation_ml_model(path_csv_compras,Departamento,Ciiu,data_ciiu_Sector,data_ventas,pipeline,models):
  """
  DESCRIPTION: This function serves for do a complementary products recomendation using the ML model.
  ------------------------------------------------------------------------------------------------------
  ------------------------------------------------------------------------------------------------------
  PARAMETERS:
  - path_csv (str): Path of the csv file with the information of the sale.
  - Departamento (str): Departamento of the customer.
  - Ciiu (int): Ciiu of the Customer.
  - data_ciiu_sector (DataFrame): DataFrame with the relation between CIIU and Sector, SubSector.
  - data_ventas (DataFrame): DataFrame with the historical sales of the company.
  ------------------------------------------------------------------------------------------------------
  ------------------------------------------------------------------------------------------------------
  RETURN:
  - articulos_recomendados (DataFrame): DataFrame with the codes and the respective names of the recomemended products.

  """
  try:
    ventas=pd.read_csv(path_csv_compras,sep=";")
    ventas.columns=["ARTICULOS_COMPRADOS","NIT"]
  except:
    ventas=pd.read_csv(path_csv_compras,sep=",")
    ventas.columns=["ARTICULOS_COMPRADOS","NIT"]

  nit=int(ventas["NIT"].loc[0])
  Departamento=Departamento.upper()
  Departamento=Departamento.replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace("Ú","U")
  Ciiu=str(Ciiu)
  try:
    Sector=data_ciiu_Sector[data_ciiu_Sector["Ciiu"]==Ciiu]["Sector"].unique()[0]
    Subsector=data_ciiu_Sector[data_ciiu_Sector["Ciiu"]==Ciiu]["SubSector"].unique()[0]
  except IndexError:
    Ciiu="4752"
    Sector=data_ciiu_Sector[data_ciiu_Sector["Ciiu"]==Ciiu]["Sector"].unique()[0]
    Subsector=data_ciiu_Sector[data_ciiu_Sector["Ciiu"]==Ciiu]["SubSector"].unique()[0]

  articulos_recomendados=[]
  for i in ventas.index:
    codigo_product=ventas["ARTICULOS_COMPRADOS"].loc[i]
    try:
      recomend_per_product=get_recommendation(str(codigo_product),Departamento,Sector,Subsector,pipeline,models)
      articulos_recomendados=articulos_recomendados+list(recomend_per_product.index)
    except KeyError:
      recomend_per_product=[]
      articulos_recomendados=articulos_recomendados+recomend_per_product
  articulos_recomendados=list(set(articulos_recomendados))
  articulos_recomendados=[int(x) for x in articulos_recomendados]

  articulos_recomendados=recomend_name_products_for_csv(data_ventas,articulos_recomendados)
  if articulos_recomendados.shape[0]==0:
    articulos_recomendados=pd.DataFrame({"Observacion":["No hay suficiente historial de ventas para hacer una recomendación de prodcutos complementarios"]})
  articulos_recomendados.to_csv("data/Articulos_recomedados.csv",sep=";",index=False)
  return articulos_recomendados

# Lista para almacenar los campos ingresados por el usuario
campos_ingresados = []

#Historical Data Ventas 2020 to 2023 Only Comercio and Only Miscelanea:
data_ventas=pd.read_csv("data/data_ventas_full_for_algoritms.csv",sep=";")

#Historical Data Ventas 2020 to 2023 Only Comercio and Only Miscelanea with Dates:
data_ventas_with_dates=pd.read_csv("data/ventas_2020_to_2023_with_dates_only_comercio_miscelanea.csv",sep=";")
#Correct the name of the Departamentos:
data_ventas_with_dates["n_departamento"]=data_ventas_with_dates["n_departamento"].str.upper()
#Correct The name of the Departamentos:
for i in data_ventas_with_dates.index:
  if data_ventas_with_dates["n_departamento"].loc[i]=="NARIÃ\x91O": #Correct NARIÑO
    data_ventas_with_dates["n_departamento"].loc[i]="NARIÑO"
  elif data_ventas_with_dates["n_departamento"].loc[i]=="BOGOTA D.C": #BOGOTA IS NOT A DEPARTAMENTO
    data_ventas_with_dates["n_departamento"].loc[i]="CUNDINAMARCA"
  elif data_ventas_with_dates["n_departamento"].loc[i]=='SAN ANDRES,PROVIDENCIA Y SANTA CATALINA': #THIS IS NOT A DEPARTAMENTO
    data_ventas_with_dates["n_departamento"].loc[i]="SAN ANDRES"

data_ventas_with_dates["nit"]=data_ventas_with_dates["nit"].astype(str)
for i in data_ventas_with_dates.index:
  data_ventas_with_dates["nit"].loc[i]=data_ventas_with_dates["nit"].loc[i].replace(".0","")
data_ventas_with_dates["nit"]=data_ventas_with_dates["nit"].astype(int)

@app.post("/recomendar-productos")
def recomendar_productos(request_data: RecomendarProductosRequest):
    
    # Agregar los campos ingresados a la lista
    campos_ingresados.append({
        "Nit del Cliente": request_data.nit_del_cliente,
        "Departamento": request_data.departamento,
        "Ciiu": request_data.ciiu,
        "Seccion": request_data.seccion,
        "Producto 1":  request_data.producto_1,
        "Producto 2":  request_data.producto_2,
        "Producto 3":request_data.producto_3,
        "Producto 4":request_data.producto_4,
        "Producto 5":request_data.producto_5,
        "Producto 6":request_data.producto_6,
        "Producto 7":request_data.producto_7,
        "Producto 8":request_data.producto_8,
        "Producto 9":request_data.producto_9,
        "Producto 10":request_data.producto_10,
        "Producto 11":request_data.producto_11,
        "Producto 12":request_data.producto_12,
        "Producto 13":request_data.producto_13,
        "Producto 14":request_data.producto_14,
        "Producto 15":request_data.producto_15,
    })
    
    p="Producto "
    df=pd.DataFrame({"ARTICULOS_COMPRADOS":[campos_ingresados[-1][p+"1"],campos_ingresados[-1][p+"2"],campos_ingresados[-1][p+"3"],
                                            campos_ingresados[-1][p+"4"],campos_ingresados[-1][p+"5"],campos_ingresados[-1][p+"6"],
                                            campos_ingresados[-1][p+"7"],campos_ingresados[-1][p+"8"],campos_ingresados[-1][p+"9"],
                                            campos_ingresados[-1][p+"10"],campos_ingresados[-1][p+"11"],campos_ingresados[-1][p+"12"],
                                            campos_ingresados[-1][p+"13"],campos_ingresados[-1][p+"14"],campos_ingresados[-1][p+"15"]],"NIT":[campos_ingresados[-1]["Nit del Cliente"],None,None,None,None,None,None,None,None,None,None,None,None,None,None]})
    df.to_csv("data/compra.csv",index=False,sep=";")
    
    path_csv_compras="data/compra.csv"
    
    #DEPARTAMENTO_CIIU:
    try:
        ventas=pd.read_csv(path_csv_compras,sep=";")
        nit=int(ventas["NIT"].loc[0])
        if data_ventas[data_ventas["nit"]==nit].shape[0]!=0:
            Departamento=data_ventas[data_ventas["nit"]==nit]["n_departamento"].unique()[0]
            Ciiu=int(data_ventas[data_ventas["nit"]==nit]["Ciiu"].unique()[0])
        elif (campos_ingresados[-1]["Departamento"]!=None and campos_ingresados[-1]["Ciiu"]!=None):
            Departamento=campos_ingresados[-1]["Departamento"]
            Ciiu=campos_ingresados[-1]["Ciiu"]
        else:
          Departamento="CUNDINAMARCA"
          Ciiu="4752"

        Departamento=Departamento.upper()
        Departamento=Departamento.replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace('Ú','U')
        
        #tipyfing for region:
        regions={"ANDINA":["ANTIOQUIA","BOYACA","CALDAS","CUNDINAMARCA","HUILA","NORTE DE SANTANDER","QUINDIO","RISARALDA","SANTANDER","TOLIMA"],
                "AMAZONIA":["AMAZONAS","CAQUETA","GUAINIA","GUAVIARE","PUTUMAYO"],
                "PACIFICA":["VALLE DEL CAUCA","CHOCO","CAUCA","NARIÑO"],
                "CARIBE":["ATLANTICO","BOLIVAR","CESAR","CORDOBA","LA GUAJIRA","MAGDALENA","SUCRE","SAN ANDRES"],
                "ORINOQUIA":["ARAUCA","CASANARE","META","VICHADA"]}
        for i in regions.keys():
            if Departamento in regions[i]:
                region=i   
            else:
              region="ANDINA"
        try:
            seccion=data_ventas[data_ventas["Ciiu"]==int(Ciiu)]["seccion"].unique()[0]
        except IndexError:
            seccion=campos_ingresados[-1]["Seccion"]
            seccion=seccion.upper()
            seccion=seccion.replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace('Ú','U')
            
        data_ventas["Ciiu"]=data_ventas["Ciiu"].astype(str)
        possible_combinations=tipificasion("n_departamento","Ciiu",data_ventas)
        articulos_recomendados=generate_recomendation(possible_combinations,data_ventas,"DEPARTAMENTO_CIIU",Departamento,Ciiu,ventas,region,seccion)
        articulos_recomendados.iloc[0]
        print("DEPTO_CIIU")
    except (KeyError, IndexError, AttributeError):
        #REGION_CIIU
        try:
            data_ventas["Ciiu"]=data_ventas["Ciiu"].astype(str)
            possible_combinations=tipificasion("REGION","Ciiu",data_ventas)
            articulos_recomendados=generate_recomendation(possible_combinations,data_ventas,"REGION_CIIU",Departamento,Ciiu,ventas,region,seccion)
            articulos_recomendados.iloc[0]
            print("REGION_CIIU")
        except (KeyError, IndexError):
            #DEPARTAMENTO_SECCION
            try:
                data_ventas["Ciiu"]=data_ventas["Ciiu"].astype(str)
                possible_combinations=tipificasion("n_departamento","seccion",data_ventas)
                articulos_recomendados=generate_recomendation(possible_combinations,data_ventas,"DEPARTAMENTO_SECCION",Departamento,Ciiu,ventas,region,seccion)
                articulos_recomendados.iloc[0]
                print("DEPTO_SECCION")
            except (KeyError, IndexError):
                #REGION_SECCION:
                try:
                    data_ventas["Ciiu"]=data_ventas["Ciiu"].astype(str)
                    possible_combinations=tipificasion("REGION","seccion",data_ventas)
                    articulos_recomendados=generate_recomendation(possible_combinations,data_ventas,"REGION_SECCION",Departamento,Ciiu,ventas,region,seccion)
                    articulos_recomendados.iloc[0]
                    print("REGION_SECCION")
                except (KeyError, IndexError):
                    
                    print("ML MODEL")
                    # Pipelien and Model:
                    pipeline = joblib.load('models/pipeline.joblib')
                    models = joblib.load("models/group_models.joblib")

                    #Data relation between  CIIU and Sector, SubSector:
                    data_ciiu_Sector=pd.read_excel("data/CIIU-Sector.xlsx")
                    articulos_recomendados=generate_recomendation_ml_model(path_csv_compras,Departamento,Ciiu,data_ciiu_Sector,data_ventas,pipeline,models)
                    print("ML MODEL")
    #Prepedido:
    prepedido=preproduct_recommend(path_csv_compras,data_ventas_with_dates)
        
    return articulos_recomendados.to_dict(orient="records"), prepedido.to_dict(orient="records")
