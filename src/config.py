
import os
from typing import Dict, List

# -------------------------------------------------------------------------
# Environment Variables & Constraints
# -------------------------------------------------------------------------
# Required Environment Variables
ENV_VARS = [
    "AZURE_STORAGE_ACCOUNT_URL",
    "AZURE_STORAGE_CONTAINER_UPLD",
    "AZURE_STORAGE_CONTAINER_WNLD",
    "AZURE_STORAGE_CONN_STR",
]

# Optional Environment Variables
OPTIONAL_ENV_VARS = [
    "AZURE_BLOB_NAME",
    "AZURE_BLOB"
]

# File Size Limit (12 MB in bytes)
MAX_FILE_SIZE_BYTES = 12 * 1024 * 1024  # 12,582,912 bytes

# -------------------------------------------------------------------------
# Column Mappings (Source Header -> Target Key)
# -------------------------------------------------------------------------
# This mapping defines the strict schema. 
# Any column in the source CSV not in this keys list will be dropped (or ignored).
COLUMN_MAPPING: Dict[str, str] = {
     'Job No.': 'job_no',
     'carr_eqp_uid': 'carr_eqp_uid',
     'Container Number': 'container_number',
     'Container Type': 'container_type',
     'Destination Service': 'destination_service',
     'Consignee Code (Multiple)': 'consignee_raw',
     'PO Number (Multiple)': 'po_numbers',
     'Booking Number (Multiple)': 'booking_numbers',
     'FCR Number (Multiple)': 'fcr_numbers',
     'Ocean BL No (Multiple)': 'obl_nos',
     'Load Port': 'load_port',
     'Final Load Port': 'final_load_port',
     'Discharge Port': 'discharge_port',
     'Last CY Location': 'last_cy_location',
     'Place of Receipt': 'place_of_receipt',
     'Place of Delivery': 'place_of_delivery',
     'Final Destination': 'final_destination',
     'First Vessel Code': 'first_vessel_code',
     'First Vessel Name': 'first_vessel_name',
     'First Voyage code': 'first_voyage_code',
     'Final Carrier Code': 'final_carrier_code',
     'Final Carrier SCAC Code': 'final_carrier_scac_code',
     'Final Carrier Name': 'final_carrier_name',
     'Final Vessel Code': 'final_vessel_code',
     'Final Vessel Name': 'final_vessel_name',
     'Final Voyage code': 'final_voyage_code',
     'True Carrier Code': 'true_carrier_code',
     'True Carrier SCAC Code': 'true_carrier_scac_code',
     'True Carrier SCAC Name': 'true_carrier_scac_name',
     'ETD LP': 'etd_lp_date',
     'ETD FLP': 'etd_flp_date',
     'ETA DP': 'eta_dp_date',
     'ETA FD': 'eta_fd_date',
     'Revised ETA': 'revised_eta_date',
     'Predictive ETA': 'predictive_eta_date',
     'ATD LP': 'atd_lp_date',
     'ATA FLP': 'ata_flp_date',
     'ATD FLP': 'atd_flp_date',
     'ATA DP': 'ata_dp_date',
     'Derived ATA DP': 'derived_ata_dp_date',
     'Revised ETA FD': 'revised_eta_fd_date',
     'Predictive ETA FD': 'predictive_eta_fd_date',
     'Cargo Received Date (Multiple)': 'cargo_receiveds_date',
     'Detention Free Days': 'detention_free_days',
     'Demurrage Free Days': 'demurrage_free_days',
     'Hot Container Flag': 'hot_container_flag',
     'Supplier/Vendor Name': 'supplier_vendor_name',
     'Manufacturer Name': 'manufacturer_name',
     'Ship To Party Name': 'ship_to_party_name',
     'Booking Approval Status': 'booking_approval_status',
     'Service Contract Number': 'service_contract_number',
     'CARRIER VEHICLE LOAD Date': 'carrier_vehicle_load_date',
     'Carrier Vehicle Load Lcn': 'carrier_vehicle_load_lcn',
     'Vehicle Departure Date': 'vehicle_departure_date',
     'Vehicle Departure Lcn': 'vehicle_departure_lcn',
     'Vehicle Arrival Date': 'vehicle_arrival_date',
     'Vehicle Arrival Lcn': 'vehicle_arrival_lcn',
     'Carrier Vehicle Unload Date': 'carrier_vehicle_unload_date',
     'Carrier Vehicle Unload Lcn': 'carrier_vehicle_unload_lcn',
     'Out Gate Date From DP': 'out_gate_from_dp_date',
     'Out Gate Location': 'out_gate_from_dp_lcn',
     'Equipment Arrived at Last CY': 'equipment_arrived_at_last_cy_date',
     'Equipment Arrival at Last Lcn': 'equipment_arrived_at_last_cy_lcn',
     'Out gate at Last CY': 'out_gate_at_last_cy_date',
     'Out gate at Last CY Lcn': 'out_gate_at_last_cy_lcn',
     'Delivery Date To Consignee': 'delivery_to_consignee_date',
     'Delivery Date To Consignee Lcn': 'delivery_to_consignee_lcn',
     'Empty Container Return Date': 'empty_container_return_date',
     'Empty Container Return Lcn': 'empty_container_return_lcn',
     'Late Booking Status': 'late_booking_status',
     'Current Departure status': 'current_departure_status',
     'Current Arrival status': 'current_arrival_status',
     'Late Arrival status': 'late_arrival_status',
     'Late Container Return status': 'late_container_return_status',
     'CO2 Emission For Tank On Wheel': 'co2_tank_on_wheel',
     'CO2 Emission For Well To Wheel': 'co2_well_to_wheel',
     'Job Type': 'job_type',
     'MCS HBL': 'mcs_hbl',
     'Transport Mode': 'transport_mode',
     'Rail Load DP Date': 'rail_load_dp_date',
     'Rail Load DP Lcn': 'rail_load_dp_lcn',
     'Rail Departure DP Date': 'rail_departure_dp_date',
     'Rail Departure DP Lcn': 'rail_departure_dp_lcn',
     'Rail Arrival Destination Date': 'rail_arrival_destination_date',
     'Rail Arrival Destination Lcn': 'rail_arrival_destination_lcn',
     'Cargo Ready Date': 'cargo_ready_date',
     'IN-DC DATE': 'in-dc_date',
     'Cargo Weight': 'cargo_weight_kg',
     'Cargo Meassure': 'cargo_measure_cubic_meter',
     'Cargo Count': 'cargo_count',
     'Cargo UM': 'cargo_um',
     'Cargo Detail Count': 'cargo_detail_count',
     'Detail Cargo UM': 'detail_cargo_um',
     '856 Filing Status': '856_filing_status',
     'Get_ISF_submission_dt': 'get_isf_submission_date',
     'Seal Number': 'seal_number',
     'In Gate Date': 'in_gate_date',
     'In Gate Lcn': 'in_gate_lcn',
     'Empty Container Dispatch Date': 'empty_container_dispatch_date',
     'Empty Container Dispatch Lcn': 'empty_container_dispatch_lcn'
}

# Date columns to enforce strict 'YYYY-MM-DD' format
DATE_COLUMNS = [v for k, v in COLUMN_MAPPING.items() if v.endswith('_date') or v in {'in-dc_date'}]

# Columns containing multiple values (dates)
MULTI_VALUED_DATE_COLS = {
    'cargo_receiveds_date',
}

# Columns containing multiple values (IDs) - to be split into lists
MULTI_VALUED_ID_COLS = [
    'po_numbers',
    'booking_numbers',
    'fcr_numbers',
    'obl_nos'
]

# Boolean flag columns
BOOLEAN_COLUMNS = [
    'hot_container_flag'
]

# ID Columns that must be strings
ID_COLUMNS = [
    "job_no", 
    "carr_eqp_uid",
    "container_number"
]

# Columns to dropped explicitly during cleanup (if not already filtered by projection)
DROP_COLUMNS_AFTER_LOAD = [
    "first_vessel_code", "first_voyage_code", "final_carrier_code", "final_carrier_scac_code",
    "final_vessel_code", "final_voyage_code", "true_carrier_code", "true_carrier_scac_code",
    "late_booking_status", "current_departure_status", "current_arrival_status",
    "late_arrival_status", "late_container_return_status"
]
