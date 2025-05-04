from sklearn.utils import all_estimators

catalog = { e[0]: e[1] for e in all_estimators(type_filter='transformer') }