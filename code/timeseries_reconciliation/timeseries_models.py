from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.model.seq2seq import MQRNNEstimator
from gluonts.mx.model.tft import TemporalFusionTransformerEstimator
from gluonts.mx.model.deepstate import DeepStateEstimator
from gluonts.mx.model.deep_factor import DeepFactorEstimator
from gluonts.mx.model.wavenet import WaveNetEstimator
from gluonts.mx.model.seq2seq import MQCNNEstimator
from gluonts.mx.model.n_beats import NBEATSEstimator
from gluonts.mx.model.transformer import TransformerEstimator


def create_estimator(level_idx, train_dataset, estimator_name):
    estimators = {
        "DeepAR": DeepAREstimator(
            freq="D",
            context_length=28,
            prediction_length=28,
            use_feat_dynamic_real=True,
            use_feat_static_cat=(level_idx != 1),
            cardinality=[len(train_dataset)] if level_idx != 1 else None,
        ),
        "MQRNN": MQRNNEstimator(
            freq="D",
            context_length=28,
            prediction_length=28,
        ),
        "TFT": TemporalFusionTransformerEstimator(
            freq="D",
            context_length=28,
            prediction_length=28,
            dynamic_feature_dims={
                "sales_mean": 1,
                "sales_std": 1,
                "sales_max": 1,
                "sales_min": 1,
                "sales_diff_mean": 1,
                "sales_lag1_mean": 1,
                "sales_lag7_mean": 1,
                "sales_lag28_mean": 1,
                "sales_rolling7_diff_mean": 1,
                "sales_rolling28_diff_mean": 1,
                "release_mean": 1,
                "out_of_stock_mean": 1,
                "sell_price_mean": 1,
                "sell_price_std": 1,
                "sell_price_max": 1,
                "sell_price_min": 1,
                "sell_price_diff_mean": 1,
                "sell_price_lag_mean": 1,
                "sell_price_rolling_mean": 1,
                "sell_price_rolling_diff_mean": 1,
                "sell_price_in_store_mean": 1,
                "year_delta": 1,
                "quarter_sin": 1,
                "quarter_cos": 1,
                "month_sin": 1,
                "month_cos": 1,
                "day_sin": 1,
                "day_cos": 1,
                "weekday_sin": 1,
                "weekday_cos": 1,
                "event_count": 1,
            },
            dynamic_cardinalities={
                "snap_CA": 2,
                "snap_TX": 2,
                "snap_WI": 2,
                "event_name_1_enc": 31,
                "event_name_2_enc": 31,
                "event_type_1_enc": 5,
                "event_type_2_enc": 5
            },
            static_cardinalities={"id": len(train_dataset)},
        ),
        "DeepState": DeepStateEstimator(
            freq="D",
            past_length=28,
            prediction_length=28,
            use_feat_dynamic_real=True,
            use_feat_static_cat=(level_idx != 1),
            cardinality=[len(train_dataset)],
        ),
        "DeepFactor": DeepFactorEstimator(
            freq="D",
            context_length=28,
            prediction_length=28,
            cardinality=[len(train_dataset)],
        ),
        "WaveNet": WaveNetEstimator(
            freq="D",
            prediction_length=28,
            cardinality=[len(train_dataset)],
        ),
        "MQCNN": MQCNNEstimator(
            freq="D",
            context_length=28,
            prediction_length=28,
            use_feat_dynamic_real=True,
            use_feat_static_cat=True,
            cardinality=[len(train_dataset)],
        ),
        "NBEATS": NBEATSEstimator(
            freq="D",
            context_length=28,
            prediction_length=28,
        ),
        "Transformer": TransformerEstimator(
            freq="D",
            context_length=28,
            prediction_length=28,
            use_feat_dynamic_real=True,
            use_feat_static_cat=True,
            cardinality=[len(train_dataset)],
        )
    }
    return estimators[estimator_name]