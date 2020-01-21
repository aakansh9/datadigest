import pyspark.sql.functions as F
import pyspark.sql.window as W
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pandas as pd
import logging
from pyspark.sql import DataFrame as SparkDataFrame


def __getspark():
    return SparkSession.builder.getOrCreate()


def __getsc():
    return SparkContext.getOrCreate()


def infer_spark_dtype(df, col):
    """
    Deduce correct spark dtype from pandas dtype for column col of pandas dataframe df
    """

    logger = logging.getLogger(__name__ + ".infer_spark_dtype")

    pd_dtype = df.dtypes[col]

    # get a sample from column col
    sample = df[col].dropna()

    if sample.shape[0] == 0:
        logger.warning("column %s of dtype %s containing nulls found" % (col, pd_dtype))
        sample = None
    else:
        sample = sample.iloc[0]

    # infer spark dtype

    # datetimes
    if pd.api.types.is_datetime64_any_dtype(pd_dtype):
        ret = T.TimestampType()

    # ints
    elif (pd_dtype == 'int8') or (pd_dtype == 'int16'):  # int8, int16
        ret = T.ShortType()

    elif pd_dtype == 'int32':
        ret = T.IntegerType()

    elif pd.api.types.is_int64_dtype(pd_dtype):
        ret = T.LongType()

    # uints
    elif pd_dtype == 'uint8':
        ret = T.ShortType()

    elif pd_dtype == 'uint16':
        ret = T.IntegerType()

    elif pd_dtype == 'uint32':
        ret = T.LongType()

    elif pd_dtype == 'uint64':
        logger.warning("converting column %s of type uint64 to spark LongType - overflows will be nulls" % col)
        ret = T.LongType()

    # floats
    elif (pd_dtype == 'float16') or (pd_dtype == 'float32'):
        ret = T.FloatType()

    elif pd_dtype == 'float64':  # float64
        ret = T.DoubleType()

    elif pd_dtype == 'bool':
        ret = T.BooleanType()

    # object
    elif pd_dtype == 'object':

        if (sample is None) or (isinstance(sample, str)):
            logger.warning("converting column %s of type object to spark StringType" % col)
            ret = T.StringType()

        elif isinstance(sample, tuple):
            raise NotImplementedError("cannot convert column %s containing tuples to spark" % col)

        else:
            raise NotImplementedError("values in column %s of type object not understood" % col)

    # category
    elif pd.api.types.is_categorical_dtype(pd_dtype):
        logger.warning("converting column %s of type category to spark StringType" % col)
        ret = T.StringType()

    else:
        raise NotImplementedError("column %s of type %s not understood" % (col, pd_dtype))

    return ret


def pd2spark(df):
    logger = logging.getLogger(__name__ + '.pd2spark')
    spark = __getspark()

    # get schema
    columns = list(df.columns)
    pd_dtypes = list(df.dtypes)
    struct_list = []

    # some pandas data types cannot be handled in spark
    conv = {columns[i]: 'float64' for i, d in enumerate(pd_dtypes) if d == 'float128'}
    if len(conv):
        logger.warning("columns %s of type float128 will be converted to float64 first" % list(conv.keys()))
        df = df.astype(conv)

    # get spark schema
    for col, pd_dtype in zip(columns, pd_dtypes):
        spark_dtype = infer_spark_dtype(df, col)
        v = T.StructField(col, spark_dtype)
        struct_list.append(v)

    scm = T.StructType(struct_list)

    # create spark df
    ret = spark.createDataFrame(df, schema=scm)

    # handle nulls in str columns

    return ret
