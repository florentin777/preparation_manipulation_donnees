"""Fonctions de la séance 05 : dates et anomalies des capteurs."""

import pandas as pd


def parse_dates(series, min_success_rate=0.95):
    """Convertit des dates et renvoie (dates_converties, rapport)."""
    converted = pd.to_datetime(
        series, format="mixed", dayfirst=True, errors="coerce"
    )

    n_total = len(series)
    n_already_missing = int(series.isna().sum())
    n_failed = int((series.notna() & converted.isna()).sum())
    n_present = n_total - n_already_missing
    success_rate = (n_present - n_failed) / n_present if n_present else 0.0

    report = {
        "n_total": n_total,
        "n_already_missing": n_already_missing,
        "n_failed": n_failed,
        "success_rate": round(success_rate, 4),
        "min_date": converted.min(),
        "max_date": converted.max(),
    }

    if success_rate < min_success_rate:
        raise ValueError(
            f"Taux de conversion insuffisant : {success_rate:.2%} "
            f"(minimum : {min_success_rate:.2%})"
        )

    return converted, report


def detect_sensor_outages(sensors, sensor_id="CAP-B", min_hours=24):
    """Détecte les suites de plus de min_hours heures sans relevé."""
    data = sensors.loc[
        sensors["id_capteur"] == sensor_id, ["horodatage"]
    ].copy()
    data["horodatage"] = pd.to_datetime(data["horodatage"], errors="coerce")
    data = data.dropna(subset=["horodatage"]).drop_duplicates()
    data = data.sort_values("horodatage")

    columns = ["debut", "fin", "duree_heures"]
    if data.empty:
        return pd.DataFrame(columns=columns)

    heures = pd.date_range(
        data["horodatage"].min(), data["horodatage"].max(), freq="h"
    )
    presence = pd.Series(True, index=data["horodatage"])
    missing = presence.reindex(heures).isna()
    groupes = missing.ne(missing.shift()).cumsum()

    pannes = []
    for _, groupe in missing.groupby(groupes):
        if groupe.iloc[0] and len(groupe) > min_hours:
            pannes.append({
                "debut": groupe.index[0],
                "fin": groupe.index[-1],
                "duree_heures": len(groupe),
            })

    return pd.DataFrame(pannes, columns=columns)


def detect_stuck_sensor(sensors, sensor_id="CAP-C", min_readings=24):
    """Détecte les suites de plus de min_readings relevés horaires constants."""
    data = sensors.loc[
        sensors["id_capteur"] == sensor_id,
        ["horodatage", "temperature_c"],
    ].copy()
    data["horodatage"] = pd.to_datetime(data["horodatage"], errors="coerce")
    data = data.dropna(subset=["horodatage"])
    data = data.sort_values("horodatage").drop_duplicates("horodatage")

    columns = ["debut", "fin", "temperature", "nombre_releves"]
    if data.empty:
        return pd.DataFrame(columns=columns)

    changements = (
        data["temperature_c"].ne(data["temperature_c"].shift())
        | data["horodatage"].diff().ne(pd.Timedelta(hours=1))
    )
    data["groupe"] = changements.cumsum()

    blocages = []
    for _, groupe in data.groupby("groupe"):
        if len(groupe) > min_readings and groupe["temperature_c"].notna().all():
            blocages.append({
                "debut": groupe["horodatage"].iloc[0],
                "fin": groupe["horodatage"].iloc[-1],
                "temperature": groupe["temperature_c"].iloc[0],
                "nombre_releves": len(groupe),
            })

    return pd.DataFrame(blocages, columns=columns)
