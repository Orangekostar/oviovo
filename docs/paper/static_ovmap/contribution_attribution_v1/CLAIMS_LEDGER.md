# Claim ledger

{
  "status": "COMPLETE",
  "modules": [
    {
      "module": "E_SOURCE",
      "effectiveness": "POSITIVE_OBSERVED",
      "attribution": "PACKAGE_ONLY",
      "novelty": "UNRESOLVED",
      "primary_cells": [
        {
          "cohort": "replica8",
          "delta": {
            "ap25": 0.03389207283292106,
            "ap50": 0.030183470726253464,
            "apall": 0.03114794302223814,
            "macc": 0.04758415288683554,
            "miou": 0.02450898026785714
          },
          "pair": [
            "A0",
            "A1"
          ]
        },
        {
          "cohort": "scannet_cf18",
          "delta": {
            "ap25": 0.03776889703803543,
            "ap50": 0.02241404468665506,
            "apall": 0.011272064253034647,
            "macc": 0.03019367936718348,
            "miou": 0.026152298615643166
          },
          "pair": [
            "A0",
            "A1"
          ]
        }
      ],
      "specific_cells": [],
      "shortlisted": false,
      "boundary": "Heterogeneous evidence increment; no Q_GAIN versus random query control.",
      "external_sources": [
        "L01",
        "L02",
        "L05"
      ],
      "independent_confirmation": "NOT_RUN_BY_CONTRACT",
      "submission_evidence_complete": false
    },
    {
      "module": "E_GROUP",
      "effectiveness": "MIXED",
      "attribution": "MATCHED_CONTRAST",
      "novelty": "KNOWN_EQUIVALENT",
      "primary_cells": [
        {
          "cohort": "replica8",
          "delta": {
            "ap25": 0.002461233654124706,
            "ap50": 0.0017213562222243295,
            "apall": 0.013345520318658821,
            "macc": 0.008029042307467094,
            "miou": -0.0028931536041464367
          },
          "pair": [
            "E6",
            "E7"
          ]
        },
        {
          "cohort": "scannet_cf18",
          "delta": {
            "ap25": 0.00898407210451746,
            "ap50": 0.014812857068254581,
            "apall": 0.005805808058743811,
            "macc": 0.00992485411588917,
            "miou": 0.004073150778354018
          },
          "pair": [
            "E6",
            "E7"
          ]
        }
      ],
      "specific_cells": [
        {
          "cohort": "replica8",
          "comparability": "SAME_FIXED_SOURCE_FUSION",
          "control": "E8",
          "delta": 0.004752699782213696,
          "method": "E7",
          "metric": "apall"
        },
        {
          "cohort": "scannet_cf18",
          "comparability": "SAME_FIXED_SOURCE_FUSION",
          "control": "E8",
          "delta": 0.007545204631098049,
          "method": "E7",
          "metric": "apall"
        }
      ],
      "shortlisted": true,
      "boundary": "Complete N/Q/F D2 is ordinary 1/4,1/4,1/2 weighting; missing-source and empty-source rules differ.",
      "external_sources": [
        "L01",
        "L05"
      ],
      "independent_confirmation": "NOT_RUN_BY_CONTRACT",
      "submission_evidence_complete": false
    },
    {
      "module": "R",
      "effectiveness": "MIXED",
      "attribution": "CONDITIONALLY_MATCHED",
      "novelty": "UNRESOLVED",
      "primary_cells": [
        {
          "cohort": "replica8",
          "delta": {
            "ap25": 0.017779637896825407,
            "ap50": 0.01561879960317461,
            "apall": 0.006450307264109348,
            "macc": 0.009757250319262889,
            "miou": 0.005814474507105316
          },
          "pair": [
            "A1",
            "A3"
          ]
        },
        {
          "cohort": "scannet_cf18",
          "delta": {
            "ap25": -0.002270684265913925,
            "ap50": -0.0012736943496612085,
            "apall": -0.000769903446822845,
            "macc": 0.0011489994688667071,
            "miou": 0.0002472257471772543
          },
          "pair": [
            "A1",
            "A3"
          ]
        }
      ],
      "specific_cells": [
        {
          "cohort": "replica8",
          "comparability": "FIXED_SURFACE_AND_E;DIFFERENT_RECOVERY_OBSERVATION_LIBRARY_NOT_PURE_SUBSTITUTION",
          "control": "U2",
          "delta": 0.0,
          "method": "A3",
          "metric": "apall"
        }
      ],
      "shortlisted": false,
      "boundary": "Fixed-surface residual support; recovered coverage is not equivalent to official true positives.",
      "external_sources": [
        "L01",
        "L03"
      ],
      "independent_confirmation": "NOT_RUN_BY_CONTRACT",
      "submission_evidence_complete": false
    },
    {
      "module": "S",
      "effectiveness": "MIXED",
      "attribution": "MATCHED_CONTRAST",
      "novelty": "CONCRETE_DIFFERENCE_IDENTIFIED",
      "primary_cells": [
        {
          "cohort": "replica8",
          "delta": {
            "ap25": 0.011775133725517528,
            "ap50": -0.004152824593718907,
            "apall": 0.00039212360013982095,
            "macc": 0.006688108464337872,
            "miou": 0.003944950648836876
          },
          "pair": [
            "FG0_FC8",
            "FG2_SURFACE"
          ]
        },
        {
          "cohort": "replica8",
          "delta": {
            "ap25": -0.002215862283022396,
            "ap50": -0.012962899549627205,
            "apall": -0.006728993879773951,
            "macc": -0.0057945538093301385,
            "miou": -0.004593785712888376
          },
          "pair": [
            "FG0_FC8",
            "S29_FG2_SURFACE"
          ]
        },
        {
          "cohort": "scannet_cf18",
          "delta": {
            "ap25": 0.0023942820126028885,
            "ap50": 0.0034694153963100993,
            "apall": 0.0019285138093638327,
            "macc": 0.003044499397818534,
            "miou": 0.003919276699974411
          },
          "pair": [
            "FG0_FC8",
            "FG2_SURFACE"
          ]
        },
        {
          "cohort": "scannet_cf18",
          "delta": {
            "ap25": 0.006730386432017121,
            "ap50": 0.006778107896717456,
            "apall": 0.002685827571267882,
            "macc": 0.006099810412908857,
            "miou": 0.006638459186630474
          },
          "pair": [
            "FG0_FC8",
            "S29_FG2_SURFACE"
          ]
        }
      ],
      "specific_cells": [
        {
          "comparability": "MATCHED_STEP_INITIALIZATION_DRAW_SCHEDULE_FULL200",
          "delta": 0.07199074074074074,
          "metric": "DEV_A_NOT_MAP_AP",
          "seed": 17,
          "step": 1000
        },
        {
          "comparability": "MATCHED_STEP_INITIALIZATION_DRAW_SCHEDULE_FULL200",
          "delta": 0.08565867003367,
          "metric": "DEV_A_NOT_MAP_AP",
          "seed": 29,
          "step": 1000
        }
      ],
      "shortlisted": true,
      "boundary": "Matched-step view versus surface grouping supports structure in DEV; selected map increment remains separate; OOD shuffle is dependence evidence only.",
      "external_sources": [
        "L02",
        "L03",
        "L04",
        "L06"
      ],
      "independent_confirmation": "NOT_RUN_BY_CONTRACT",
      "submission_evidence_complete": false
    }
  ],
  "same_step_recognition": [
    {
      "arrays": {
        "bytes": 976903,
        "path": "/mnt/shared/capacity/node101/ww/ovimap-frozen-fc-routing-v1/attempt_001/training/seed17/FG1_VIEW/dev_1000.npz",
        "sha256": "010f83dd01d50a10bc5148643fce3f772c543053e12cf77f626e28a8440992cf"
      },
      "method": "FG1_VIEW",
      "metrics": {
        "A": 0.3887205387205387,
        "C": 0.40808080808080816,
        "CE": 3.911798397712535,
        "M": 0.5743777056277056,
        "NLL": 4.1399633715223985,
        "categories": 28,
        "corrections": 22,
        "families": 4,
        "full200_competitors": true,
        "harm": 17,
        "net_corrections": 5,
        "original_objects": 69,
        "per_family_A": {
          "scene0105": 0.46969696969696967,
          "scene0143": 0.4629629629629629,
          "scene0211": 0.39999999999999997,
          "scene0620": 0.2222222222222222
        },
        "variant_records": 276
      },
      "receipt": "c51939600895620fcb5bacd6b554884e13ad114f41e39b70c39c068228abf2d5",
      "seed": 17,
      "status": "COMPLETE",
      "step": 1000,
      "weight_at_step1000_available": "NOT_INFERRED_FROM_RECORDS"
    },
    {
      "arrays": {
        "bytes": 976348,
        "path": "/mnt/shared/capacity/node101/ww/ovimap-frozen-fc-routing-v1/attempt_001/training/seed17/FG2_SURFACE/dev_1000.npz",
        "sha256": "dbeb598095330d1e6d17a208cae2c7e5de32cf1d3d866e13db0cb314e0555d23"
      },
      "method": "FG2_SURFACE",
      "metrics": {
        "A": 0.46071127946127943,
        "C": 0.4801136363636363,
        "CE": 3.949017267634069,
        "M": 0.58866341991342,
        "NLL": 4.176042225753719,
        "categories": 28,
        "corrections": 24,
        "families": 4,
        "full200_competitors": true,
        "harm": 9,
        "net_corrections": 15,
        "original_objects": 69,
        "per_family_A": {
          "scene0105": 0.46969696969696967,
          "scene0143": 0.6481481481481481,
          "scene0211": 0.39166666666666666,
          "scene0620": 0.33333333333333326
        },
        "variant_records": 276
      },
      "receipt": "cdab0d89d191c6ede818fb5efbc52690fe940786563cbc135479f5efa963001e",
      "seed": 17,
      "status": "COMPLETE",
      "step": 1000,
      "weight_at_step1000_available": "NOT_INFERRED_FROM_RECORDS"
    },
    {
      "arrays": {
        "bytes": 976834,
        "path": "/mnt/shared/capacity/node101/ww/ovimap-frozen-fc-routing-v1/attempt_001/training/seed29/FG1_VIEW/dev_1000.npz",
        "sha256": "c3e4fe3c5674822d2c8c9cf1b5055b29a360a93cf699e048d16737248d5f7cca"
      },
      "method": "FG1_VIEW",
      "metrics": {
        "A": 0.3760732323232323,
        "C": 0.3752525252525253,
        "CE": 3.923835598612173,
        "M": 0.551758658008658,
        "NLL": 4.151832248385907,
        "categories": 28,
        "corrections": 24,
        "families": 4,
        "full200_competitors": true,
        "harm": 18,
        "net_corrections": 6,
        "original_objects": 69,
        "per_family_A": {
          "scene0105": 0.4848484848484849,
          "scene0143": 0.3333333333333333,
          "scene0211": 0.4083333333333334,
          "scene0620": 0.2777777777777778
        },
        "variant_records": 276
      },
      "receipt": "2beac18d8c41be15b9748d6606cb8ec0f2d59636e4215b0a3bff85ba675320e7",
      "seed": 29,
      "status": "COMPLETE",
      "step": 1000,
      "weight_at_step1000_available": "NOT_INFERRED_FROM_RECORDS"
    },
    {
      "arrays": {
        "bytes": 976526,
        "path": "/mnt/shared/capacity/node101/ww/ovimap-frozen-fc-routing-v1/attempt_001/training/seed29/FG2_SURFACE/dev_1000.npz",
        "sha256": "40a519b724f427e1204ec05662b226770e09796f4f54277b02649df0318af45d"
      },
      "method": "FG2_SURFACE",
      "metrics": {
        "A": 0.4617319023569023,
        "C": 0.46363636363636357,
        "CE": 3.9442955003342646,
        "M": 0.5967983405483405,
        "NLL": 4.171117485460107,
        "categories": 28,
        "corrections": 25,
        "families": 4,
        "full200_competitors": true,
        "harm": 8,
        "net_corrections": 17,
        "original_objects": 69,
        "per_family_A": {
          "scene0105": 0.4242424242424242,
          "scene0143": 0.6666666666666666,
          "scene0211": 0.4041666666666666,
          "scene0620": 0.35185185185185186
        },
        "variant_records": 276
      },
      "receipt": "84b324b4d3807c3845e5a3e25878787b44d81eb4c72d40423a43a360bd0ccf42",
      "seed": 29,
      "status": "COMPLETE",
      "step": 1000,
      "weight_at_step1000_available": "NOT_INFERRED_FROM_RECORDS"
    }
  ],
  "scope": "EXPOSED_EXPLORATORY_NOT_CONFIRMATORY",
  "core_or_CVPR_proven_claim": false,
  "identity": "3bd2246751d29af13f980f2440236b192aa547d4bfd3f1995e0f20a8dbe2fecc"
}

具体近邻：[原始来源与区别表](../../../../artifacts/static_ovmap/contribution_attribution_v1/records/literature.json.gz)。
