

numerical_cols = ['condition', 'car_age', 'mileage']
categorical_cols = ['brand', 'market_model', 'body_type', 'transmission', 'state', 'interior', 'color']

market_map = {
    "Economy Sedan": [
        "Altima", "Focus", "Impala", "Sonata", "Cruze", "Taurus",
        "Optima", "200", "Avenger", "Passat", "Civic", "Corolla",
        "Fusion", "Malibu", "Sentra", "Elantra", "Jetta", "Accord",
        "Camry", "Versa", "Rio", "Yaris", "Forte", "Fiesta", "Sonic", "Cobalt", "Sebring"],

    "Luxury Sedan": [
        "3 Series", "5 Series", "7 Series", "A4", "A6", "A8", "C-Class",
        "E-Class", "S-Class", "S60", "S90", "TLX", "Q50", "Genesis",
        "Maxima", "300", "G Sedan", "LS", "ES", "IS"],

    "Sports Sedan": [
        "M3", "M5", "Charger", "S4", "CTS-V", "IS F", "XFR", "WRX"],

    "Economy SUV": [
        "Explorer", "Edge", "Journey", "Escape", "Rogue", "Tucson",
        "Equinox", "Sorento", "CX-5", "CR-V", "RAV4", "Highlander",
        "Santa Fe", "Kicks", "HR-V", "Sportage", "Soul", "Pilot", "Pathfinder", "Traverse", "Durango"],

    "Luxury SUV": [
        "Grand Cherokee", "X5", "X3", "Q7", "Q5", "GLC", "GLA",
        "RX", "LX", "Cayenne", "MDX", "Escalade", "Navigator",
        "GLE", "Macan", "XC90", "Tahoe", "Suburban", "Murano", "Expedition"],

    "Off-Road SUV": [
        "Wrangler", "4Runner", "Bronco", "Defender", "G-Class"],

    "Pickup Truck": [
        "1500", "F-150", "Silverado 1500", "Ram Pickup 1500",
        "Tacoma", "Tundra", "Ranger", "Colorado", "Frontier", "F-250 Super Duty", "Silverado 2500HD"],

    "Electric Vehicle": [
        "Leaf", "Model S", "Model 3", "Model X", "Model Y",
        "Bolt", "i3", "i4", "Polestar 2", "Mach-E"],

    "Sports Car": [
        "Mustang", "Camaro", "Corvette", "370Z", "911", "M4",
        "Supra", "GT-R", "F-Type", "718 Cayman"],

    "Minivan": [
        "Sienna", "Odyssey", "Grand Caravan", "Town and Country",
        "Quest", "Pacifica", "Sedona"],

    "Hybrid Car": [
        "Prius", "Camry Hybrid", "Accord Hybrid", "Fusion Hybrid",
        "Highlander Hybrid", "RAV4 Hybrid"],

    "Economy Hatchback": [
        "Caliber", "PT Cruiser", "Accent"],
}


