
DROP TABLE IF EXISTS fact_sales CASCADE;
DROP TABLE IF EXISTS dim_product CASCADE;
DROP TABLE IF EXISTS dim_geography CASCADE;
DROP TABLE IF EXISTS dim_client_sale CASCADE;

CREATE TABLE dim_product (
    product_key INTEGER PRIMARY KEY,         
    product VARCHAR(100) NOT NULL,
    product_type VARCHAR(100) NOT NULL
);
COMMENT ON TABLE dim_product IS 'Contiene la lista única de los 8 productos y sus categorías estrictas.';


CREATE TABLE dim_geography (
    city_key INTEGER PRIMARY KEY,           
    city VARCHAR(100) NOT NULL,
    country VARCHAR(100) NOT NULL
);
COMMENT ON TABLE dim_geography IS 'Contiene las ciudades y países únicos de las transacciones.';

CREATE TABLE dim_client_sale (
    client_sale_key INTEGER PRIMARY KEY,    
    client_type VARCHAR(50) NOT NULL,
    sale_type VARCHAR(50) NOT NULL
);
COMMENT ON TABLE dim_client_sale IS 'Contiene la combinación única del tipo de cliente y el canal de venta.';


CREATE TABLE fact_sales (
    sale_id BIGSERIAL PRIMARY KEY, 
    date DATE NOT NULL,
    product_key INTEGER NOT NULL,
    city_key INTEGER NOT NULL,
    client_sale_key INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(10, 2) NOT NULL,      
    discount NUMERIC(5, 2) NOT NULL,         
    shipping_cost NUMERIC(10, 2) NOT NULL,
    total NUMERIC(12, 2) NOT NULL,         

    CONSTRAINT fk_product
        FOREIGN KEY (product_key)
        REFERENCES dim_product (product_key),
    
    CONSTRAINT fk_city
        FOREIGN KEY (city_key)
        REFERENCES dim_geography (city_key),
    
    CONSTRAINT fk_client_sale
        FOREIGN KEY (client_sale_key)
        REFERENCES dim_client_sale (client_sale_key)
);
COMMENT ON TABLE fact_sales IS 'Contiene todos los hechos de ventas, referenciando las dimensiones.';