import pandas as pd
import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.window as W
import pyspark.sql.types as T
from pyspark.sql import DataFrame as SparkDataFrame
import logging, logging.config


def pretty_name(x):
    x *= 100
    if x == int(x):
        return '%.0f%%' % x
    else:
        return '%.1f%%' % x


def describe_integer_1d(df, column):
    logger = logging.getLogger(__name__ + '.describe_integer_1d')

    if isinstance(df, SparkDataFrame):

        dtype = df.select(column).dtypes[0][1]
        assert dtype in ['tinyint', 'smallint', 'bigint']

        nrows = df.count()

        stats = (df
                 .select(column)
                 .filter(~F.isnull(column))
                 .filter(~F.isnan(column))
                 .agg(F.mean(F.col(column)).alias("mean"),
                      F.min(F.col(column)).alias("min"),
                      F.max(F.col(column)).alias("max"),
                      F.variance(F.col(column)).alias("variance"),
                      F.stddev(F.col(column)).alias("std"),
                      F.sum(F.col(column)).alias("sum"),
                      F.count(F.col(column) == 0.0).alias('n_zeros'),
                      F.countDistinct(column).alias("non_missing_distinct_count"),

                      F.kurtosis(F.col(column)).alias("kurtosis"),
                      F.skewness(F.col(column)).alias("skewness"),
                      F.count(column).alias("non_missing_count"),
                      )
                 .toPandas()
                 )

        # approx percentiles
        for x in [0.05, 0.25, 0.5, 0.75, 0.95]:
            stats[pretty_name(x)] = (df.select(column)
                                         .na.drop()
                                         .selectExpr("percentile(`{col}`,CAST({n} AS DOUBLE))"
                                                     .format(col=column, n=x))
                                         .toPandas().iloc[:, 0]
                                         )

        stats = stats.iloc[0]
        stats.name = column

        stats['dtype'] = dtype
        stats['nrows'] = nrows

        stats["range"] = stats["max"] - stats["min"]
        stats["iqr"] = stats[pretty_name(0.75)] - stats[pretty_name(0.25)]
        stats["cv"] = stats["std"] / float(stats["mean"])

        stats["mad"] = (df
                        .select(column)
                        .filter(~F.isnull(column))
                        .filter(~F.isnan(column))
                        .select(F.abs(F.col(column) - stats["mean"]).alias("delta"))
                        .agg(F.sum(F.col("delta")))
                        .toPandas().iloc[0, 0] / float(stats["non_missing_count"]))

        stats['p_zeros'] = stats['n_zeros'] / float(nrows)
        stats['n_missing'] = stats['nrows'] - stats['non_missing_count']
        stats['p_missing'] = 1 - (stats['non_missing_count'] / float(stats['nrows']))
        stats['p_unique'] = stats['non_missing_distinct_count'] / float(stats['non_missing_count'])

    else:
        raise NotImplementedError("df not understood")

    return stats


def describe_float_1d(df, column):
    logger = logging.getLogger(__name__ + '.describe_float_1d')

    if isinstance(df, SparkDataFrame):

        dtype = df.select(column).dtypes[0][1]
        assert (dtype in ['float', 'double']) or ('decimal' in dtype)

        nrows = df.count()

        stats = (df
                 .select(column)
                 .filter(~F.isnull(column))
                 .filter(~F.isnan(column))
                 .agg(F.mean(F.col(column)).alias("mean"),
                      F.min(F.col(column)).alias("min"),
                      F.max(F.col(column)).alias("max"),
                      F.variance(F.col(column)).alias("variance"),
                      F.stddev(F.col(column)).alias("std"),
                      F.sum(F.col(column)).alias("sum"),
                      F.count(F.col(column) == 0.0).alias('n_zeros'),
                      F.countDistinct(column).alias("non_missing_distinct_count"),

                      F.kurtosis(F.col(column)).alias("kurtosis"),
                      F.skewness(F.col(column)).alias("skewness"),
                      F.count(column).alias("non_missing_count"),
                      )
                 .toPandas()
                 )

        for x in [0.05, 0.25, 0.5, 0.75, 0.95]:
            stats[pretty_name(x)] = (df.select(column)
                                         .na.drop()
                                         .selectExpr("percentile_approx(`{col}`,CAST({n} AS DOUBLE))"
                                                     .format(col=column, n=x)).toPandas().iloc[:, 0]
                                         )
        stats = stats.iloc[0]
        stats.name = column

        stats['dtype'] = dtype
        stats['nrows'] = nrows

        stats["range"] = stats["max"] - stats["min"]
        stats["iqr"] = stats[pretty_name(0.75)] - stats[pretty_name(0.25)]
        stats["cv"] = stats["std"] / float(stats["mean"])

        stats["mad"] = (df
                        .select(column)
                        .filter(~F.isnull(column))
                        .filter(~F.isnan(column))
                        .select(F.abs(F.col(column) - stats["mean"]).alias("delta"))
                        .agg(F.sum(F.col("delta")))
                        .toPandas().iloc[0, 0] / float(stats["non_missing_count"]))

        stats['p_zeros'] = stats['n_zeros'] / float(nrows)
        stats['n_missing'] = stats['nrows'] - stats['non_missing_count']
        stats['p_missing'] = 1 - (stats['non_missing_count'] / float(stats['nrows']))
        stats['p_unique'] = stats['non_missing_distinct_count'] / float(stats['non_missing_count'])

    else:
        raise NotImplementedError("df not understood")

    return stats


def describe_string_1d(df, column):
    logger = logging.getLogger(__name__ + '.describe_string_1d')

    if isinstance(df, SparkDataFrame):

        dtype = df.select(column).dtypes[0][1]
        assert dtype == 'string'

        nrows = df.count()

        stats = (df
                 .select(column)
                 .filter(~F.isnull(column))
                 .filter(~F.isnan(column))
                 .agg(F.min(F.col(column)).alias("min"),
                      F.max(F.col(column)).alias("max"),
                      F.countDistinct(column).alias("non_missing_distinct_count"),
                      F.count(column).alias("non_missing_count"),
                      )
                 .toPandas()
                 )

        for x in [0.05, 0.25, 0.5, 0.75, 0.95]:
            stats[pretty_name(x)] = (df.select(column)
                                         .na.drop()
                                         .selectExpr("percentile_approx(`{col}`,CAST({n} AS DOUBLE))"
                                                     .format(col=column, n=x)).toPandas().iloc[:, 0]
                                         )

        stats = stats.iloc[0]
        stats.name = column

        stats['dtype'] = dtype
        stats['nrows'] = nrows

        stats['n_missing'] = stats['nrows'] - stats['non_missing_count']
        stats['p_missing'] = 1 - (stats['non_missing_count'] / float(stats['nrows']))
        stats['p_unique'] = stats['non_missing_distinct_count'] / float(stats['non_missing_count'])

    else:
        raise NotImplementedError("df not understood")

    return stats


def describe_datetime_1d(df, column):
    logger = logging.getLogger(__name__ + '.describe_datetime_1d')

    if isinstance(df, SparkDataFrame):

        dtype = df.select(column).dtypes[0][1]
        assert dtype in ['timestamp', 'date']

        nrows = df.count()

        stats = (df
                 .select(column)
                 .filter(~F.isnull(column))
                 .agg(F.min(F.col(column)).alias("min"),
                      F.max(F.col(column)).alias("max"),
                      F.count(column).alias("non_missing_count"),
                      F.countDistinct(column).alias("non_missing_distinct_count"))
                 .toPandas())

        stats = stats.iloc[0]
        stats.name = column

        stats['dtype'] = dtype
        stats['nrows'] = nrows

        stats['n_missing'] = stats['nrows'] - stats['non_missing_count']
        stats['p_missing'] = 1 - (stats['non_missing_count'] / float(stats['nrows']))
        stats['p_unique'] = stats['non_missing_distinct_count'] / float(stats['non_missing_count'])

        # Convert Pandas timestamp object to regular datetime:
        if isinstance(stats["max"], pd.Timestamp):
            stats = stats.astype(object)
            stats["max"] = str(stats["max"].to_pydatetime())
            stats["min"] = str(stats["min"].to_pydatetime())

        stats["range"] = stats["max"] - stats["min"]

    else:
        raise NotImplementedError("df not understood")

    return stats


def describe_1d(df, column):
    if isinstance(df, SparkDataFrame):

        dtype = df.select(column).dtypes[0][1]

        if ("array" in dtype) or ("stuct" in dtype) or ("map" in dtype):
            raise NotImplementedError("Column {c} of type {t} cannot be analyzed".format(c=column, t=dtype))

        elif dtype in ['tinyint', 'smallint', 'bigint']:
            stats = describe_integer_1d(df, column)

        elif (dtype in ['float', 'double']) or ('decimal' in dtype):
            stats = describe_float_1d(df, column)

        elif dtype in ['timestamp', 'date']:
            stats = describe_datetime_1d(df, column)

        elif dtype == 'string':
            stats = describe_string_1d(df, column)

        elif dtype == 'boolean':
            raise NotImplementedError("Column {c} of type {t} cannot be analyzed".format(c=column, t=dtype))

        else:
            raise NotImplementedError("Column {c} of type {t} cannot be analyzed".format(c=column, t=dtype))

    else:
        raise NotImplementedError("df not understood")

    return stats


def describe(df):
    for col in df.columns:
        describe_1d(df, col)

    return stats
