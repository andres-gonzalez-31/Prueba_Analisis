import pandas as pd
import numpy as np
import re
from datetime import datetime
import unicodedata
import os
from typing import Dict, List, Any, Optional

# --- 1. DATA CLEANER CLASS ---

class AutomatedDataCleaner:
    """
    Sistema de Limpieza de Datos automatizado y reutilizable.
    Implementa la eliminación estricta de filas con datos faltantes.
    """
    
    def __init__(self, csv_file_path: str, cleaning_config: Optional[Dict] = None):
        """Inicializa con la ruta del archivo CSV."""
        self.csv_file_path = csv_file_path
        self.df = None
        self.cleaning_config = cleaning_config or self._default_configuration()
        self.cleaning_statistics = {}
        
    def _default_configuration(self) -> Dict:
        """Configuración por defecto para reglas de limpieza y mapeo de columnas."""
        return {
            'column_mapping': {
                'date': ['fecha', 'date', 'fecha_venta', 'fecha_compra', 'timestamp'],
                'product': ['producto', 'product', 'item', 'articulo', 'descripcion'],
                'product_type': ['tipo_producto', 'categoria', 'categoria_producto', 'tipo', 'category', 'tipo_producto'],
                'quantity': ['cantidad', 'qty', 'quantity', 'unidades', 'units'],
                'unit_price': ['precio_unitario', 'precio', 'price', 'unit_price', 'precio_unidad'],
                'total': ['total_ventas', 'venta_total', 'total', 'amount', 'monto_total', 'importe'],
                'sale_type': ['tipo_venta', 'canal_venta', 'channel', 'venta_tipo', 'sales_channel'],
                'client_type': ['tipo_cliente', 'cliente_tipo', 'customer_type', 'segmento_cliente'],
                'discount': ['descuento', 'discount', 'descuento_porcentaje', 'discount_percent'],
                'shipping_cost': ['costo_envio', 'shipping_cost', 'costo_transporte', 'envio_costo'],
                'city': ['ciudad', 'city', 'localidad', 'municipio'],
                'country': ['pais', 'country', 'paises', 'nation'],
            },
            'cleaning_rules': {
                'text': {'min_length': 1, 'max_length': 100, 'case': 'title'},
                'number': {'min_value': 0, 'max_value': 1000000, 'decimals': 2},
                'date': {'formats': ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S'], 'min_range': '2020-01-01', 'max_range': '2025-12-31'}
            },
            'preferred_column_order': [
                'date', 'product', 'product_type', 'quantity', 'unit_price',
                'city', 'country', 'sale_type', 'client_type', 'discount', 'shipping_cost', 'total'
            ]
        }
    
    def _reorganize_malformed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorganiza datos si parecen cargados en columnas incorrectas."""
        if len(df.columns) >= 11 and 'total' not in df.columns:
            try:
                new_data = {
                    'city': df.iloc[:, 0].astype(str), 'date': df.iloc[:, 1].astype(str),
                    'product': df.iloc[:, 2].astype(str), 'product_type': df.iloc[:, 3].astype(str),
                    'quantity': df.iloc[:, 4].astype(str), 'unit_price': df.iloc[:, 5].astype(str),
                    'sale_type': df.iloc[:, 6].astype(str), 'client_type': df.iloc[:, 7].astype(str),
                    'discount': df.iloc[:, 8].astype(str), 'shipping_cost': df.iloc[:, 9].astype(str),
                    'total': df.iloc[:, 10].astype(str)
                }
                df_corrected = pd.DataFrame(new_data)
                return df_corrected
            except Exception:
                return df
        return df

    def _normalize_text_for_comparison(self, text: str) -> str:
        """Normaliza texto (minúsculas, sin acentos, sin caracteres especiales) para comparación."""
        if pd.isna(text): return ""
        text = str(text).lower().strip()
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        text = re.sub(r'[^a-z0-9]', '', text)
        return text

    def _correct_product_type_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Corrige el tipo de producto basado en el nombre del producto,
        mapeando ESTRICTAMENTE a los 8 productos deseados.
        """
        if 'product' not in df.columns:
            return df
            
        df['product_clean'] = df['product'].fillna('').astype(str).apply(self._normalize_text_for_comparison)
        
        # --- Mapeo ESTRICTO para los 8 productos ---
        # El mapeo se hace por palabra clave limpia (e.g., 'leche' -> 'Lácteo')
        product_to_type_map = {
            'arepa': 'Alimento_Percedero',
            'pan': 'Alimento_Percedero', 
            'leche': 'Lácteo', 
            'queso': 'Lácteo', 
            'yogurt': 'Lácteo', 
            'te': 'Bebida', 
            'cafe': 'Bebida', 
            'chocolate': 'Bebida', 
        }

        # Aplicar mapeo. Si el producto limpio está en el mapa, asignar su tipo.
        # Si no está, el tipo de producto existente se mantiene, pero será limpiado después.
        df['new_product_type'] = df['product_clean'].map(product_to_type_map)
        
        # Unificar con la columna original (si new_product_type es NaN, usa product_type original)
        df['product_type'] = df['new_product_type'].fillna(df['product_type'])
        
        # Limpieza final de strings
        df['product_type'] = df['product_type'].astype(str).str.title().str.strip().replace('Nan', np.nan)
        
        # Limpiar y normalizar el nombre del producto en base a los 8 esperados (ej: 'Leches' -> 'Leche')
        def clean_product_name(text):
            norm_text = self._normalize_text_for_comparison(text)
            for key in product_to_type_map.keys():
                # Si el nombre normalizado del producto contiene una de las claves
                if key in norm_text:
                    return key.title() # Retorna el nombre limpio (e.g., 'Leche')
            return text # Si no se encuentra, dejar el nombre original (será limpiado en batch cleaning)

        df['product'] = df['product'].apply(clean_product_name)

        df.drop(columns=['product_clean', 'new_product_type'], inplace=True, errors='ignore')
        return df

    def _detect_and_correct_country(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta y corrige el país basándose en la ciudad. Los no encontrados se marcan como NaN."""
        city_country_map = {
            'bogota': 'Colombia', 'medellin': 'Colombia', 'cali': 'Colombia', 'barranquilla': 'Colombia',
            'cartagena': 'Colombia', 'cucuta': 'Colombia', 'bucaramanga': 'Colombia', 'pereira': 'Colombia',
            'antofagasta': 'Chile', 'santiago': 'Chile', 'valparaiso': 'Chile', 'concepcion': 'Chile',
            'viña del mar': 'Chile', 'lima': 'Perú', 'cusco': 'Perú', 'arequipa': 'Perú',
            'ciudad de mexico': 'México', 'monterrey': 'México', 'guadalajara': 'México',
            'madrid': 'España', 'sevilla': 'España', 'córdoba': 'España', 'barcelona': 'España',
            'buenos aires': 'Argentina', 'cordoba': 'Argentina', 'rosario': 'Argentina'
        }
        
        if 'city' in df.columns:
            cities_clean = df['city'].fillna('').astype(str).str.lower().str.strip()
            
            def find_country(city):
                city = self._normalize_text_for_comparison(city) 
                for city_map, country in city_country_map.items():
                    if self._normalize_text_for_comparison(city_map) == city:
                        return country
                return np.nan # Retornar NaN en lugar de 'Unknown'
            
            df['country'] = cities_clean.apply(find_country)
            df['country'].replace('Unknown', np.nan, inplace=True) 
        return df
    
    def _automatic_column_mapping(self, original_columns: List[str]) -> Dict:
        """Mapea automáticamente los nombres de columna originales a nombres estándar (Fallback)."""
        mapping = {}
        columns_mapped = set()
        columns_not_mapped = original_columns.copy()
        
        columns_normalized = {col: self._normalize_text_for_comparison(col) for col in original_columns}
        variants_normalized = {}
        
        for standard_name, variants in self.cleaning_config['column_mapping'].items():
            variants_normalized[standard_name] = [self._normalize_text_for_comparison(v) for v in variants]
        
        # Primer paso: Coincidencias exactas
        for col_original, col_normalized in columns_normalized.items():
            if col_original in columns_mapped: continue
            for standard_name, variants in variants_normalized.items():
                if col_normalized in variants:
                    mapping[col_original] = standard_name
                    columns_mapped.add(col_original)
                    if col_original in columns_not_mapped: columns_not_mapped.remove(col_original)
                    break
        # ... El resto de la lógica de mapeo automático sigue aquí ...
        
        # Columnas no mapeadas: Asignar un nombre genérico
        standard_names_used = set(mapping.values())
        for i, column in enumerate(columns_not_mapped):
            standard_name = f"unmapped_col_{i+1}"
            while standard_name in standard_names_used:
                i += 1
                standard_name = f"unmapped_col_{i+1}"
            mapping[column] = standard_name
            standard_names_used.add(standard_name)
        
        return mapping

    def _drop_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina columnas con nombres duplicados."""
        columns_seen = set()
        columns_to_drop = []
        for column in df.columns:
            if column in columns_seen:
                columns_to_drop.append(column)
            else:
                columns_seen.add(column)
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        return df

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reordena las columnas según el orden preferido."""
        current_columns = df.columns.tolist()
        preferred_order = self.cleaning_config.get('preferred_column_order', [])
        ordered_cols = [col for col in preferred_order if col in current_columns]
        remaining_cols = [col for col in current_columns if col not in ordered_cols]
        final_columns = ordered_cols + sorted(remaining_cols)
        if final_columns != current_columns:
            df = df[final_columns]
        return df

    def _apply_batch_cleaning(self):
        """Aplica la conversión de tipos y reglas de limpieza en un proceso por lotes."""
        numeric_cols_map = {'quantity': 'int64', 'unit_price': 'float64', 'discount': 'float64', 'shipping_cost': 'float64', 'total': 'float64'}
        for col in numeric_cols_map:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce') 
                if col in ['quantity', 'discount', 'shipping_cost']: 
                    self.df[col] = self.df[col].fillna(0)
                    
                min_val = self.cleaning_config['cleaning_rules']['number']['min_value']
                max_val = self.cleaning_config['cleaning_rules']['number']['max_value']
                decimals = self.cleaning_config['cleaning_rules']['number']['decimals']
                self.df[col] = self.df[col].clip(lower=min_val, upper=max_val)
                if decimals is not None: self.df[col] = self.df[col].round(decimals)
                
        if 'date' in self.df.columns:
            # Intentar convertir las fechas, si falla se convierte en NaT
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce', format='mixed')
            
        text_cols = [col for col in self.df.columns if col not in list(numeric_cols_map.keys()) and col != 'date']
        if text_cols: 
            self.df[text_cols] = self.df[text_cols].apply(self._vectorized_text_cleaning)

    def _vectorized_text_cleaning(self, series: pd.Series) -> pd.Series:
        """Aplica limpieza de texto (quitar espacios, caracteres especiales, mayúsculas/minúsculas)."""
        series_clean = series.astype(str).str.strip()
        series_clean = series_clean.str.replace(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑüÜ\s\-_\.]', ' ', regex=True)
        series_clean = series_clean.str.replace(r'\s+', ' ', regex=True).str.strip()
        series_clean = series_clean.replace('', np.nan) 
        config_case = self.cleaning_config['cleaning_rules']['text']['case']
        if config_case == 'title': series_clean = series_clean.str.title()
        return series_clean
    
    def _calculate_clean_total(self):
        """Calcula o limpia la columna 'total'."""
        required_cols = ['quantity', 'unit_price', 'discount', 'shipping_cost']
        if all(col in self.df.columns and self.df[col].dtype.kind in 'fi' for col in required_cols):
            base_revenue = self.df['quantity'] * self.df['unit_price']
            # Asumiendo que 'discount' es un porcentaje (e.g., 0.10)
            discount_factor = self.df['discount']
            net_revenue = base_revenue * (1 - discount_factor)
            shipping_cost = self.df['shipping_cost']
            
            valid_prices = self.df['unit_price'].notna()
            self.df.loc[valid_prices, 'total'] = (net_revenue[valid_prices] + shipping_cost[valid_prices]).round(self.cleaning_config['cleaning_rules']['number']['decimals'])

    def _strict_final_cleanup(self):
        """
        ### PASO CRÍTICO SOLICITADO
        Realiza la eliminación estricta de filas con valores 'Nan', 'Unknown', 
        o cualquier valor nulo (NaN de NumPy) en CUALQUIER columna.
        """
        initial_rows = len(self.df)
        
        # 1. Reemplazar strings problemáticos con np.nan para que dropna los detecte
        self.df.replace(['Nan', 'Unknown', 'Null', 'None', ''], np.nan, inplace=True, regex=True)
        
        # 2. ELIMINAR CUALQUIER FILA QUE CONTENGA AL MENOS UN VALOR NULO
        self.df.dropna(how='any', inplace=True)
        
        rows_removed = initial_rows - len(self.df)
        print(f"    STRICT CLEANUP: Removed {rows_removed:,} rows due to NaN/Unknown/Null values in any column.")
        
    def _calculate_cleaning_statistics(self, original_records: int):
        """Calcula y almacena estadísticas de limpieza."""
        nulls_per_column = self.df.isnull().sum()
        total_cells_final = len(self.df) * len(self.df.columns)
        self.cleaning_statistics = {
            'original_records': original_records,
            'final_records': len(self.df),
            'final_columns': len(self.df.columns),
            'nulls_per_column': nulls_per_column.to_dict(),
            'records_removed': original_records - len(self.df),
            'completeness_percentage': (1 - nulls_per_column.sum() / total_cells_final) * 100 if total_cells_final > 0 else 100
        }
    
    def _display_cleaning_summary(self):
        """Imprime un resumen de los resultados de la limpieza."""
        stats = self.cleaning_statistics
        print("\n" + "="*60)
        print("DATA CLEANING SUMMARY")
        print("="*60)
        print(f"Original Records: {stats['original_records']:,}")
        print(f"Final Records: {stats['final_records']:,}")
        print(f"Records Removed: {stats['records_removed']:,}")
        print(f"Overall Completeness: {stats['completeness_percentage']:.1f}%")
        
    def save_cleaned_data(self, output_file_path: str = "cleaned_data.csv"):
        """Guarda el DataFrame limpio en un archivo CSV."""
        self.df = self._drop_duplicate_columns(self.df) 
        
        # Último chequeo de nulos antes de guardar
        self.df.replace(['Nan', 'Unknown', 'Null', 'None', ''], np.nan, inplace=True, regex=True)
        self.df.dropna(how='any', inplace=True)
        
        if not self.df.empty:
            self.df.to_csv(output_file_path, index=False, encoding='utf-8')
            return True
        return False
        
    def apply_cleaning(self, custom_mapping: Optional[Dict] = None) -> pd.DataFrame:
        """Aplica el proceso completo de limpieza de datos."""
        try:
            self.df = pd.read_csv(self.csv_file_path, low_memory=False)
            original_records = len(self.df)
            
            self.df = self._reorganize_malformed_data(self.df)

            # --- CORRECCIÓN DE ERRORES: Mapeo de columnas explícito ---
            fixed_manual_map = {
                'Ciudad': 'city', 'Fecha': 'date', 'Producto': 'product', 
                'Tipo_Producto': 'product_type', 'Cantidad': 'quantity', 
                'Precio_Unitario': 'unit_price', 'Tipo_Venta': 'sale_type', 
                'Tipo_Cliente': 'client_type', 'Descuento': 'discount', 
                'Costo_Envio': 'shipping_cost', 'Total': 'total'
            }
            
            # Aplicar mapeo. Se usa `errors='ignore'` si la columna no existe.
            current_mapping = {k: v for k, v in fixed_manual_map.items() if k in self.df.columns}
            self.df = self.df.rename(columns=current_mapping)

            # --- FASES DE LIMPIEZA
            self.df = self._correct_product_type_mapping(self.df) # <--- AHORA MÁS ESTRICTO
            self.df = self._drop_duplicate_columns(self.df)
            self._apply_batch_cleaning()
            self.df = self._detect_and_correct_country(self.df)
            self._calculate_clean_total()
            
            # --- FASE DE ELIMINACIÓN ESTRICTA DE FILAS (SOLICITADA)
            self._strict_final_cleanup() 
            
            self.df = self._reorder_columns(self.df)
            
            self._calculate_cleaning_statistics(original_records)
            self._display_cleaning_summary()
            
            return self.df
            
        except KeyError as e:
            # Captura si alguna columna crítica no existe incluso después del mapeo
            print(f"KeyError: Columna crítica faltante o con nombre inesperado: {e}.")
            print("Asegúrate de que los encabezados del archivo 'RWventas.csv' coincidan con los esperados: Ciudad, Fecha, Producto, Tipo_Producto, etc.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error general durante el proceso de limpieza: {e}")
            return pd.DataFrame()

# ----------------------------------------------------------------------
# --- 2. NORMALIZATION FUNCTION (CON ELIMINACIÓN ESTRICTA DE NULLS) ---
# ----------------------------------------------------------------------

def normalize_sales_data(df_cleaned: pd.DataFrame):
    """
    Normaliza el DataFrame limpio en cuatro tablas relacionales (Star Schema)
    y los guarda como archivos CSV separados, garantizando que no haya nulos ni duplicados.
    """
    print("\n" + "="*60)
    print("STARTING DATA NORMALIZATION (4 Tables)")
    print("="*60)
    
    # ----------------------------------------------------
    # 1. DIMENSIÓN DE PRODUCTOS (dim_product) - AHORA FIJA
    # ----------------------------------------------------
    
    # Definir los 8 productos y categorías de forma manual
    product_data = {
        'product': ['Arepa', 'Leche', 'Queso', 'Te', 'Cafe', 'Chocolate', 'Yogurt', 'Pan'],
        'product_type': ['Alimento_Percedero', 'Lácteo', 'Lácteo', 'Bebida', 'Bebida', 'Bebida', 'Lácteo', 'Alimento_Percedero']
    }
    dim_product = pd.DataFrame(product_data)
    dim_product.drop_duplicates(subset=['product', 'product_type'], inplace=True)
    dim_product.dropna(how='any', inplace=True)
    
    dim_product['product_key'] = np.arange(len(dim_product)) + 1 
    dim_product = dim_product[['product_key', 'product', 'product_type']]
    
    print(f"    Dim_Product creado con {len(dim_product)} productos fijos.")
    # ----------------------------------------------------
    
    # 2. Dimensión Geográfica (dim_geography)
    dim_geography = df_cleaned[['city', 'country']].drop_duplicates().reset_index(drop=True)
    dim_geography['city_key'] = np.arange(len(dim_geography)) + 1
    dim_geography = dim_geography[['city_key', 'city', 'country']]
    
    # GARANTIZAR NULOS Y DUPLICADOS EN DIMENSIÓN
    dim_geography.dropna(how='any', inplace=True)
    dim_geography.drop_duplicates(subset=['city', 'country'], inplace=True)
    
    # 3. Dimensión Tipo Cliente / Venta (dim_client_sale)
    dim_client_sale = df_cleaned[['client_type', 'sale_type']].drop_duplicates().reset_index(drop=True)
    dim_client_sale['client_sale_key'] = np.arange(len(dim_client_sale)) + 1
    dim_client_sale = dim_client_sale[['client_sale_key', 'client_type', 'sale_type']]

    # GARANTIZAR NULOS Y DUPLICADOS EN DIMENSIÓN
    dim_client_sale.dropna(how='any', inplace=True)
    dim_client_sale.drop_duplicates(subset=['client_type', 'sale_type'], inplace=True)

    # 4. Tabla de Hechos (fact_sales)
    print("    Injecting Foreign Keys into Fact Table...")
    
    # Merge para inyectar product_key
    # NOTA: Solo los registros que coincidan con la dimensión FIJA obtendrán una clave.
    # Los registros de ventas con productos NO 'Arepa, Leche, etc.' obtendrán NaN y se eliminarán luego.
    df_fact = pd.merge(df_cleaned,
                       dim_product[['product', 'product_type', 'product_key']], 
                       on=['product', 'product_type'], 
                       how='left')
    
    # Merge para las otras claves
    df_fact = pd.merge(df_fact, dim_geography, on=['city', 'country'], how='left')
    df_fact = pd.merge(df_fact, dim_client_sale, on=['client_type', 'sale_type'], how='left')
    
    # ❌ ELIMINACIÓN ESTRICTA DE FILAS SIN CLAVE FORÁNEA O MÉTRICAS CRÍTICAS
    initial_fact_rows = len(df_fact)
    # Se añade la verificación de 'date' para NaT que puede ocurrir en la limpieza
    df_fact.dropna(
        subset=['product_key', 'city_key', 'client_sale_key', 'date', 'total', 'quantity'], 
        inplace=True
    )
    rows_removed_fact = initial_fact_rows - len(df_fact)
    print(f"    Removed {rows_removed_fact:,} rows from Fact Table due to missing keys/metrics (including products outside the 8 fixed ones).")

    # Limpieza de columnas de apoyo
    cols_to_drop = ['product', 'product_type', 'city', 'country', 'client_type', 'sale_type']
    df_fact.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Seleccionar columnas finales para fact_sales
    fact_sales_cols = [
        'date', 'product_key', 'city_key', 'client_sale_key', 
        'quantity', 'unit_price', 'discount', 'shipping_cost', 'total'
    ]
    fact_sales = df_fact[fact_sales_cols].copy()
    
    # Asegurar tipo entero para claves y tipo adecuado para métricas
    fact_sales['product_key'] = fact_sales['product_key'].astype('Int64') # Usar Int64 para manejar NaNs que no deberían estar, pero por seguridad
    fact_sales['city_key'] = fact_sales['city_key'].astype('Int64')
    fact_sales['client_sale_key'] = fact_sales['client_sale_key'].astype('Int64')
    
    # --- 5. GUARDAR TABLAS ---
    
    output_dir = "normalized_data"
    os.makedirs(output_dir, exist_ok=True)
    
    tables_to_save = {
        "fact_sales": fact_sales,
        "dim_product": dim_product,
        "dim_geography": dim_geography,
        "dim_client_sale": dim_client_sale
    }
    
    for table_name, df_table in tables_to_save.items():
        output_path = os.path.join(output_dir, f"{table_name}.csv")
        # ❌ Doble chequeo final y estricto: Eliminación de Nulos antes de guardar
        df_table.dropna(how='any', inplace=True) 
        df_table.to_csv(output_path, index=False, encoding='utf-8')
        print(f"    Saved table {table_name} with {len(df_table):,} rows to: {output_path}. (NULL-FREE GUARANTEED)")

# --- EXECUTION ---

if __name__ == "__main__":
    
    try:
        from dotenv import load_dotenv
        if os.path.exists(".env"):
            load_dotenv()
    except ImportError:
        pass
        
    INPUT_FILE = "RWventas.csv" 
    CLEANED_OUTPUT_FILE = "RWventas_CLEANED_STRICTLY_FINAL.csv"
    
    if os.path.exists(INPUT_FILE):
        print(f"STARTING FULL DATA PIPELINE for {INPUT_FILE}")
        
        # 1. Execute Cleaning
        cleaner = AutomatedDataCleaner(INPUT_FILE)
        df_cleaned = cleaner.apply_cleaning()
        
        if not df_cleaned.empty:
            # Save the fully cleaned CSV
            cleaner.save_cleaned_data(CLEANED_OUTPUT_FILE)
            
            # 2. Execute Normalization (Splitting into 4 tables with strict null removal)
            normalize_sales_data(df_cleaned)
            
            print("\nPROCESS COMPLETED! Check the 'normalized_data' folder for the final NULL-FREE CSV files.")
        else:
            print("\nFATAL ERROR: Cleaning produced an empty DataFrame. Halting process.")
    else:
        print(f"Error: File {INPUT_FILE} not found. Please ensure the raw data file is in the same directory.")