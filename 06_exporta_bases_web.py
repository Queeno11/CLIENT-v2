import gc
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

import utils
import test_tools
import procesa_bases

PATH = r"D:\World Bank\CLIENT v2"
DATA_RAW = rf"{PATH}\Data\Data_raw"
DATA_PROC = rf"{PATH}\Data\Data_proc"
DATA_OUT = rf"{PATH}\Data\Data_out"
GPW_PATH = rf"D:\Datasets\Gridded Population of the World"


def genera_mapa_climate_dashboard():
    ''' Genera el mapa para el Climate Dashboard.
    
    Abre las bases WB_country_IDs.feather (los IDs generados) y load_WB_country_data() para 
    verificar consistencia. 
    
    El archivo final es un archivo CSV con los nombres y sus códigos de adm0, adm1 y adm2,
    junto a su geometría.
    
    El archivo se exporta a {DATA_OUT}\\for webpage\\WB_map.csv
    
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame resultante con la información geográfica y administrativa.

    '''
    # Load ID dataset
    gdf = gpd.read_feather(rf"{DATA_PROC}\\WB_country_IDs.feather")

    # Add names from the original WB adm2 dataset
    gdf_raw = procesa_bases.load_WB_country_data()
    gdf_raw = gdf_raw[["ISO_A3", "ADM1CD_c", "ADM2CD_c", "geometry"]]
    assert gdf_raw.duplicated(subset=["ISO_A3", "ADM1CD_c", "ADM2CD_c"]).sum() == 0, "There are duplicated entries in the raw dataset!!"

    # Merge both datasets to assert that the codes are correct and consistent
    gdf = gdf.merge(gdf_raw.drop(columns="geometry"), how="outer", on=["ISO_A3", "ADM1CD_c", "ADM2CD_c"], indicator=True, validate="1:1")
    assert (gdf._merge == "both").all(), "There are problems with the merge!!"
    gdf = gdf.drop(columns="_merge")

    # Rename for export
    gdf = gdf.rename(
        columns={
            "ISO_A3": "adm0_code", 
            "ADM1CD_c": "adm1_code", 
            "ADM2CD_c": "adm2_code",
            "NAM_0": "adm0_name",
            "NAM_1": "adm1_name",
            "NAM_2": "adm2_name",
        }
    )
    print(gdf.columns)
    # Remove conflicting boundaries # FIXME: this should be implemented in the load_WB_country_data function...
    gdf = procesa_bases.fix_disputed_boundaries(gdf)
    
    # Exporta
    outname = rf"{DATA_OUT}\\for webpage\\WB_map.parquet"
    gdf.drop(columns="ID").to_parquet(outname, index=False) # Export without the ID column
    print(f"Se creó {outname}")
    
    return gdf
    
def genera_shocks_climate_dashboard(gdf):
    ''' Genera la base con la información geográfica de los shocks para el CLimate Dashboard.
    
    Para cada tipo de shock (floods, drought, hurricanes, intenserain, heatwaves, coldwaves):
      - Carga la base de datos correspondiente en formato CSV.
      - Establece los tipos de datos optimizados para la memoria.
      - Realiza un reshape (melt) de la base para convertirla a formato largo.
      - Realiza merge con la información geográfica de la base de IDs (gdf).
      - Aplica funciones de validación (a través de test_tools) para verificar la consistencia de las columnas y forma.
      - Exporta la base procesada a un archivo CSV en {DATA_OUT}\\for webpage\\WB_{shock}.csv.

    Args:
        gdf (geopandas.GeoDataFrame): Base geográfica con información de los países y códigos administrativos.

    '''
    # Set admin level to categorical dtype (when the dataset is expanded, it will be more memory efficient)
    gdf["ID"]        = gdf["ID"].astype("category")
    gdf["adm0_code"] = gdf["adm0_code"].astype("category")
    gdf["adm1_code"] = gdf["adm1_code"].astype("category")
    gdf["adm2_code"] = gdf["adm2_code"].astype("category")
    gdf = gdf.set_index("ID")
    gdf = gdf.drop(columns=["adm0_name","adm1_name","adm2_name"])
    
    # Set dtypes to make this loading efficient
    dtypes = {"year": np.int16, "variable":"category", "threshold":"category", "area_affected":np.float32, "population_affected":np.float32, "ID":np.int64}# "adm2_code": np.int16, "adm1_code": np.int16, "adm0_code": np.int16,

    for shock in ["floods", "drought", "hurricanes", "intenserain", "heatwaves", "coldwaves"]:
        print(shock)
        df = pd.read_csv(
            rf"{DATA_OUT}\\WB_{shock}_long.csv",
            dtype=dtypes, 
            usecols=dtypes.keys(),
        )
            
        # Set ID to categorical dtype (this is after loading as int to match with the categories of gdf)
        df["ID"] = df["ID"].astype("category")
        
        # Reshape to long format
        df = df.melt(id_vars=["ID", "year", "variable", "threshold"], var_name="measure", value_name="value")

        # Set categorical and index to make faster merges
        df["measure"] = df["measure"].astype("category")
        df["year"] = df["year"].astype("category")
        df = df.set_index(["ID"])    
        
        # Add adm0, adm1 and adm2 codes    
        df = gdf.drop(columns=["geometry"]).join(df, on=["ID"], how="inner", validate="1:m")
        df = df.reset_index()

        # Set index to make faster merges and expand dataset
        #   Replace columns with null categories with zeros before setting the index to make it work as expected:
        index = ["ID", "year", "variable", "threshold", "measure"]
        for col in index:
            if (df[col].dtype == "category"):
                if (df[col].cat.categories.shape[0]==0):
                    df[col] = df[col].astype(float).fillna(0)
                    df[col] = df[col].astype("category")

        df = df.set_index(index)
        df = utils.expand_dataset(df, gdf)

        # Test the output
        test_tools.assert_correct_colnames(df)
        test_tools.assert_correct_shape(df, gdf)

        # Export
        outpath = rf"{DATA_OUT}\\for webpage\\WB_{shock}.csv"
        df.to_csv(outpath, index=False)
        print(f"Se creó {outpath}")
        
def genera_mapa_hc_dashboard():
    '''Genera el mapa geográfico para el HC Dashboard.

    La función realiza los siguientes pasos:
      - Carga la base de datos de IDs de IPUMS desde un archivo Feather.
      - Carga y procesa la base de datos original de World Bank y de IPUMS.
      - Verifica la consistencia de los códigos administrativos mediante un merge.
      - Renombra columnas para adecuarlas al formato deseado.
      - Exporta el resultado final a {DATA_OUT}\\for webpage\\HC_geo_map.csv.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame completo con la información geográfica original de IPUMS (sin la columna ID).
    '''
    # Load dataset
    gdf = gpd.read_feather(rf"{DATA_PROC}\IPUMS_country_IDs.feather").drop(columns=["ID"])
    WB_country = procesa_bases.load_WB_country_data()
    IPUMS_country = procesa_bases.load_IPUMS_country_data(WB_country, keep_name=True)
    IPUMS_country = IPUMS_country.clip(WB_country.total_bounds)

    # Merge to assert that the codes are correct and consistent
    assert (gdf.merge(IPUMS_country, on=["GEOLEVEL1", "GEOLEVEL2", "CNTRY_CODE"], how="outer", indicator=True, validate="1:1")._merge == "both").all()

    # Rename columns to make it in the intended format
    IPUMS_country = IPUMS_country.rename(columns={
        "CNTRY_CODE":"adm0", 
        "GEOLEVEL1":"adm1", 
        "GEOLEVEL2":"adm2", 
        "CNTRY_NAME":"adm0_name", 
        "ADMIN_NAME":"adm2_name"
    }).drop(columns="ID")
    
    # Export to CSV
    outpath = rf"{DATA_OUT}\for webpage\HC_geo_map.parquet"
    IPUMS_country.to_parquet(outpath, index=False)
    print(f"Se creó {outpath}")
    
    return IPUMS_country
    
def genera_shocks_nacionales_hc_dashboard(gdf_full):
    '''Genera la base de datos de shocks a nivel nacional para el HC Dashboard.

    La función realiza los siguientes pasos:
      - Lista y carga todos los archivos CSV que contienen datos nacionales de HC 
        (con "HC_national_data" en el nombre), que provienen del script de STATA.
      - Agrega columnas 's3' y 's4' combinando la información de las variables s3a, 
        s3b, s3c, s3d, s3f y s4a, s4b, s4c respectivamente.
      - Realiza un merge con la base geográfica gdf_full para validar la existencia de los códigos.
      - Exporta el DataFrame final a {DATA_PROC}\\for webpage\\HC_national_data.csv.

    Args:
        gdf_full (geopandas.GeoDataFrame): Base geográfica completa con la información a nivel nacional.

    Returns:
        None
    '''
    import os
    import pandas as pd

    files = os.listdir(rf"{DATA_OUT}\HC Treatment Complete")
    files = [f for f in files if "HC_national_data" in f and f.endswith(".csv")]

    dfs = []
    for file in files:
        df = pd.read_csv(rf"{DATA_OUT}\HC Treatment Complete\{file}")
        df["s3"] = pd.NA
        df["s4"] = pd.NA

        s3cols = ["s3a", "s3b", "s3c", "s3d", "s3f"]
        s4cols = ["s4a", "s4b", "s4c"]

        for col in s3cols:
            df["s3"] = df["s3"].fillna(df[col])
            assert (df[s3cols].notna().sum(axis=1) <= 1).all(), \
                f"Multiple non-null values in s3 columns: {df[(df[s3cols].notna().sum(axis=1) > 1)]}"
        for col in s4cols:
            df["s4"] = df["s4"].fillna(df[col])
            assert (df[s4cols].notna().sum(axis=1) <= 1).all(), \
                f"Multiple non-null values in s4 columns: {df[(df[s4cols].notna().sum(axis=1) > 1)]}"

        dfs.append(df)

    df = pd.concat(dfs)
    for col in df.columns:
        assert not df[col].isna().all()

    # Drop s3* columns
    df = df.drop(columns=[col for col in df.columns if ("s3" in col or "s4" in col) and (col != "s3" and col != "s4")])
    # Order variables
    df = df[["adm0", "s1", "s2", "s3", "s4", "s5", "outcome", "new", "v", "status"]]
    df = df.rename(columns={"new":"time", "v": "value", "status":"treatment"})
    df.loc[df.s1 == "Hurricane", "s5"] = df.loc[df.s1 == "Hurricane", "s5"] / 100

    df = df.merge(gdf_full[["adm0"]].drop_duplicates(), on=["adm0"], validate="m:1")
    print(f"Hay datos de {df.adm0.unique().size} países. Exportando CSV")
    df.to_csv(rf"{DATA_OUT}\for webpage\HC_national_data.csv", index=False)
    print(rf"Se creó {DATA_OUT}\for webpage\HC_national_data.csv")
    return

def genera_shocks_subnacionales_hc_dashboard():
    '''Genera la base de datos de shocks a nivel subnacional para el HC Dashboard.

    La función realiza los siguientes pasos:
      - Lista y carga todos los archivos CSV que contienen datos geográficos de HC 
      (archivos que contienen "HC_geodata"), que provienen del script de STATA
      - Para cada archivo:
          * Agrega columnas 's3' y 's4' combinando la información de las variables s3a, 
          s3b, s3c, s3d, s3f y s4a, s4b, s4c respectivamente.
          * Elimina las columnas s3* y s4* originales y reordena las variables.
          * Realiza un merge con la base geográfica gdf_full para validar la información a nivel subnacional.
          * Valida la fusión mediante funciones de test (test_tools.validate_hc_merge).
          * Exporta o añade la información al archivo {DATA_OUT}\\for webpage\\HC_geo_data.csv.

    Args:
        gdf_full (geopandas.GeoDataFrame): Base geográfica completa con información a nivel subnacional.

    Returns:
        None
    '''
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client
        
    client = Client()

    # Shocks      
    files = os.listdir(rf"{DATA_OUT}\HC Treatment Complete")
    files = [f for f in files if "HC_geodata" in f and f.endswith(".csv")]

    # Remove previous data
    try: 
        os.remove(rf"{DATA_OUT}\for webpage\HC_geo_data.csv") 
    except: 
        pass

    dtypes = {
        "adm0":         np.int32,
        "adm1":         np.int64,
        "adm2":         np.int64,
        "s1":             object,
        "s2":             object,
        "s3a":            object,
        "s3b":            object,
        "s3c":            object,
        "s3d":            object,
        "s3f":            object,
        "s4a":            object,
        "s4b":            object,
        "s4c":            object,
        "s5":             object,
        "outcome":        object,
        "diftime":    np.float32,
        "status":         object,
    }
    
    dfs = []
    for file in files:
        
        df = dd.read_csv(rf"{DATA_OUT}\HC Treatment Complete\{file}", assume_missing=True, dtype=dtypes, blocksize="250MB")
        df = df.assign(s3=pd.NA, s4=pd.NA)
        
        s3cols = ["s3a", "s3b", "s3c", "s3d", "s3f"]
        s4cols = ["s4a", "s4b", "s4c"]

        df["s3"] = df[s3cols].apply(
            lambda row: next((x for x in row if pd.notnull(x)), pd.NA),
            axis=1,
            meta=('s3', object)
        )

        df["s4"] = df[s4cols].apply(
            lambda row: next((x for x in row if pd.notnull(x)), pd.NA),
            axis=1,
            meta=('s4', object)
        )

        # Drop s3* columns
        df = df.drop(columns=s3cols+s4cols)

        # Order variables
        df = df[["adm0", "adm1", "adm2", "s1", "s2", "s3", "s4", "s5", "outcome", "status", "diftime"]]
        df = df.rename(columns={"status":"treatment_sub", "diftime":"diff"})

        def adjust_hurricane(s1, s5):
            if s1 == "Hurricane":
                return str(float(s5) / 100)
            return s5
        df["s5"] = df.apply(lambda x: adjust_hurricane(x["s1"], x["s5"]), axis=1, meta=("object"))
           
        num_paises = df["adm0"].nunique().compute()
        print(f"Hay datos de {num_paises} países")

        dfs += [df]
    
    df = dd.concat(dfs)
    print("Exportando... Chequear el cliente de dask, tipicamente en http://localhost:8787/status")
    df.to_csv(rf"{DATA_OUT}\for webpage\HC_geo_data.csv", index=False, single_file=True)
    print("Se creó {DATA_OUT}\\for webpage\\HC_geo_data.csv")
    
    return
    
def genera_etiquetas_hc_dashboard():
    '''Genera la base de etiquetas para el HC Dashboard.

    Transforma el excel button_labels.xlsx a un archivo CSV y lo exporta a 
    {DATA_OUT}\\for webpage\\selector_labels.csv. Si hay que editar las etiquetas,
    editar el excel original.
    
    Returns:
        None
    '''
    raise Exception("Deprecated, do not use...")    
    
    df = pd.read_excel(rf"{DATA_RAW}\button_labels.xlsx")
    
    outpath = rf"{DATA_OUT}\for webpage\selector_labels.csv"
    df.to_csv(outpath, index=False)
    print(f"Se creó {outpath}")
    
    return

def genera_shocks_subnacionales_ops_dashboard():
    ''' Genera la base de datos de shocks a nivel subnacional para el OPS Dashboard.
    
    La función realiza los siguientes pasos:
        - Carga los archivos CSV que contienen datos de shocks a nivel subnacional del Climate dashboard.
        - Agrupa los datos por códigos administrativos y calcula la media de los valores.
        - Exporta el DataFrame resultante a un archivo CSV en {DATA_OUT}\\for webpage\\OPS_geo_data.csv.
    
    '''
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    
    results = []

    for shock in ["coldwaves", "heatwaves", "intenserain", "hurricanes", "drought", "floods"]:
        df = dd.read_csv(rf"{DATA_OUT}\for webpage\WB_{shock}.csv", blocksize="1GB", assume_missing=True)

        for n in [5,10,15]:
            df_sel = df[df.year > 2020-n]
            result = df_sel.groupby(["adm0_code", "adm1_code", "adm2_code", "variable", "threshold", "measure"])["value"].mean()
            result = result.reset_index()
            result["timeframes"] = n
            results += [result]

    # Concatenate
    results = dd.concat(results)

    with ProgressBar():
        results.to_csv(rf"{DATA_OUT}\for webpage\OPS_geo_data.csv", index=False, single_file=True)
    
def genera_zip():
    '''Empaqueta en un archivo ZIP todos los archivos CSV del directorio {DATA_OUT}\for webpage.'''

    import zipfile
    
    path = rf"{DATA_OUT}\for webpage"
    files = os.listdir(path)
    files = [f for f in files if f.endswith(".csv")]
    
    with zipfile.ZipFile(rf"{DATA_OUT}\for webpage\HC_data.zip", "w") as zipf:
        for file in files:
            zipf.write(rf"{path}\{file}", arcname=file)

if __name__ == "__main__":
    import argparse
        
    parser = argparse.ArgumentParser(description="Generate dashboard data")
    parser.add_argument("--db", choices=["climate", "ops", "hc", "all"], default="all", required=True, help="Choose the database to run: climate, ops, or hc")

    args = parser.parse_args()

    if (args.db == "climate") | (args.db == "all"):
        print("Generando bases del Climate Dashboard")
        gdf = genera_mapa_climate_dashboard()
        genera_shocks_climate_dashboard(gdf)
        gc.collect()
    
    if (args.db == "ops") | (args.db == "all"):
        print("Generando bases del OPS Dashboard")
        genera_shocks_subnacionales_ops_dashboard()
        gc.collect() 
    
    if (args.db == "hc") | (args.db == "all"):
        print("Generando bases del HC Dashboard")
        gdf_full = genera_mapa_hc_dashboard()
        genera_shocks_nacionales_hc_dashboard(gdf_full)
        genera_shocks_subnacionales_hc_dashboard()
        gc.collect()

    # genera_zip()
    print("Listo! Datos exportados para las páginas web.")    
    
    print("#########################################################")
    print("####      Testing the generated datasets:            ####")
    print("#########################################################")
    
    if (args.db == "climate") | (args.db == "all"):

        print("Testing CLIMATE DATASET:")
        # Load the data
        gdf = pd.read_parquet(rf"{DATA_OUT}\\for webpage\\WB_map.parquet")

        for shock in ["floods", "drought", "hurricanes", "intenserain", "heatwaves", "coldwaves"]:
            
            print("Verifying", shock)
            df = pd.read_csv(rf"{DATA_OUT}\\for webpage\\WB_{shock}.csv")
            
            test_tools.assert_correct_colnames(df)
            test_tools.assert_correct_shape(df, gdf)
            test_tools.validate_climate_dataset(df, gdf)
            if shock=="floods":
                total_chunks=8**2
            elif shock=="hurricanes":
                total_chunks=6**2
            else: 
                total_chunks=4**2
                
            test_tools.compare_xarray_with_zonal_statistics(adm="WB", chunk_number=None, shockname=shock, var=None, total_chunks=total_chunks, year=None, out_name=f"test_climate_{shock}")
    
    if (args.db == "ops") | (args.db == "all"):
        print("TO DO: Test the OPS_geo_data.csv")
    
    if (args.db == "hc") | (args.db == "all"):

        print("Testing HC DATASET:")
        
        test_tools.validate_hc_merge()
        for shock in ["floods", "drought", "hurricanes", "intenserain", "heatwaves", "coldwaves"]:
            
            if shock=="floods":
                total_chunks=8**2
            elif shock=="hurricanes":
                total_chunks=6**2
            else: 
                total_chunks=4**2
            print(total_chunks)
            test_tools.compare_xarray_with_zonal_statistics(adm="IPUMS", chunk_number=None, shockname=shock, var=None, total_chunks=total_chunks, year=None, out_name=f"test_hc_{shock}")

    print("All tests passed! Check de output images")
    