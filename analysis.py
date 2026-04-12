import pandas as pd

def load_and_process_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except:
        raise Exception("❌ Error loading data.csv")

    # Convert dates safely
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True, errors='coerce')

    # Drop only critical nulls
    df = df.dropna(subset=['Order Date', 'Ship Date', 'Product Name', 'State/Province'])

    # Lead Time
    df['Lead Time'] = (df['Ship Date'] - df['Order Date']).dt.days

    # Keep valid rows
    df = df[df['Lead Time'] >= 0]

    # Product → Factory Mapping
    product_factory_map = {
        "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
        "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
        "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
        "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
        "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
        "Laffy Taffy": "Sugar Shack",
        "SweeTARTS": "Sugar Shack",
        "Nerds": "Sugar Shack",
        "Fun Dip": "Sugar Shack",
        "Fizzy Lifting Drinks": "Sugar Shack",
        "Everlasting Gobstopper": "Secret Factory",
        "Hair Toffee": "The Other Factory",
        "Lickable Wallpaper": "Secret Factory",
        "Wonka Gum": "Secret Factory",
        "Kazookles": "The Other Factory"
    }

    df['Factory'] = df['Product Name'].map(product_factory_map)
    df['Factory'] = df['Factory'].fillna("Unknown")

    factory_coords = {
        "Lot's O' Nuts": (32.88, -111.76),
        "Wicked Choccy's": (32.07, -81.08),
        "Sugar Shack": (48.11, -96.18),
        "Secret Factory": (41.44, -90.56),
        "The Other Factory": (35.11, -89.97),
        "Unknown": (39.50, -98.35)
    }

    df['Latitude'] = df['Factory'].map(lambda x: factory_coords[x][0])
    df['Longitude'] = df['Factory'].map(lambda x: factory_coords[x][1])
    # Routes
    df['Route_State'] = df['Factory'] + " → " + df['State/Province']

    # Delay flag
    df['Delayed'] = df['Lead Time'] > 5

    # Efficiency Score (higher = better)
    df['Efficiency Score'] = 1 / (df['Lead Time'] + 1)

    return df


def route_analysis(df):
    route_df = df.groupby('Route_State').agg(
        total_shipments=('Order ID', 'count'),
        avg_lead_time=('Lead Time', 'mean'),
        std_lead_time=('Lead Time', 'std'),
        delay_rate=('Delayed', 'mean'),
        efficiency=('Efficiency Score', 'mean')
    ).reset_index()

    return route_df.sort_values(by='efficiency', ascending=False)


def ship_mode_analysis(df):
    return df.groupby('Ship Mode').agg(
        avg_lead_time=('Lead Time', 'mean'),
        total_orders=('Order ID', 'count')
    ).reset_index()
