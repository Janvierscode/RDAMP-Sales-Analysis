"""
ACE Superstore Sales Analysis
Author: Data Analyst
Date: July 4, 2025

This script performs comprehensive analysis of ACE Superstore sales data
to provide business intelligence insights for executive decision-making.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load and initial preprocessing of the datasets"""
    print("Loading datasets...")
    
    # Load main sales data with proper encoding
    try:
        sales_df = pd.read_csv('Ace Superstore Retail Dataset(in).csv', encoding='utf-8')
    except UnicodeDecodeError:
        sales_df = pd.read_csv('Ace Superstore Retail Dataset(in).csv', encoding='latin1')
    
    # Load store locations data
    try:
        stores_df = pd.read_csv('Store Locations(Store Locations).csv', encoding='utf-8')
    except UnicodeDecodeError:
        stores_df = pd.read_csv('Store Locations(Store Locations).csv', encoding='latin1')
    
    print(f"Sales data shape: {sales_df.shape}")
    print(f"Store locations shape: {stores_df.shape}")
    
    return sales_df, stores_df

def clean_and_prepare_data(sales_df, stores_df):
    """Clean and prepare data for analysis"""
    print("\nCleaning and preparing data...")
    
    # Convert date column
    sales_df['Order Date'] = pd.to_datetime(sales_df['Order Date'])
    
    # Calculate revenue (Sales - Cost)
    sales_df['Revenue'] = sales_df['Sales'] - sales_df['Cost Price']
    
    # Calculate profit margin
    sales_df['Profit Margin'] = (sales_df['Revenue'] / sales_df['Sales']) * 100
    
    # Calculate total revenue per order
    sales_df['Total Revenue'] = sales_df['Revenue'] * sales_df['Quantity']
    
    # Calculate total sales per order
    sales_df['Total Sales'] = sales_df['Sales'] * sales_df['Quantity']
    
    # Merge with store locations to get proper region mapping
    sales_df = sales_df.merge(stores_df[['City', 'Postal Code', 'Region']], 
                             on=['City', 'Postal Code'], 
                             how='left', 
                             suffixes=('', '_store'))
    
    # Use store region where available, otherwise use original region
    sales_df['Region_Final'] = sales_df['Region_store'].fillna(sales_df['Region'])
      # Create segments based on product categories
    sales_df['Segment'] = sales_df['Category'].apply(lambda x: 
        'Food & Beverages' if pd.notna(x) and x.startswith('Food') else
        'Home & Living' if pd.notna(x) and x in ['Home', 'Kitchen', 'Home Appliances'] else
        'Health & Beauty' if pd.notna(x) and x in ['Health', 'Beauty', 'Grooming'] else
        'Fashion' if pd.notna(x) and x.startswith('Clothing') else
        'Other')
    
    print(f"Data prepared. Final shape: {sales_df.shape}")
    return sales_df

def exploratory_analysis(df):
    """Perform exploratory data analysis"""
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print(f"Date range: {df['Order Date'].min()} to {df['Order Date'].max()}")
    print(f"Total orders: {len(df):,}")
    print(f"Total customers: {df['Customer ID'].nunique():,}")
    print(f"Total products: {df['Product ID'].nunique():,}")
    print(f"Total revenue: ${df['Total Revenue'].sum():,.2f}")
    print(f"Total sales: ${df['Total Sales'].sum():,.2f}")
    print(f"Average order value: ${df['Total Sales'].mean():.2f}")
    
    # Missing values
    print(f"\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    return df

def analyze_by_region_segment(df):
    """Analyze sales by region and segment"""
    print("\n" + "="*50)
    print("ANALYSIS BY REGION AND SEGMENT")
    print("="*50)
    
    # Regional analysis
    regional_summary = df.groupby('Region_Final').agg({
        'Total Sales': 'sum',
        'Total Revenue': 'sum',
        'Discount': 'mean',
        'Order ID': 'count'
    }).round(2)
    regional_summary.columns = ['Total Sales', 'Total Revenue', 'Avg Discount Rate', 'Order Count']
    regional_summary = regional_summary.sort_values('Total Revenue', ascending=False)
    
    print("REGIONAL PERFORMANCE:")
    print(regional_summary)
    
    # Segment analysis
    segment_summary = df.groupby('Segment').agg({
        'Total Sales': 'sum',
        'Total Revenue': 'sum',
        'Discount': 'mean',
        'Order ID': 'count'
    }).round(2)
    segment_summary.columns = ['Total Sales', 'Total Revenue', 'Avg Discount Rate', 'Order Count']
    segment_summary = segment_summary.sort_values('Total Revenue', ascending=False)
    
    print("\nSEGMENT PERFORMANCE:")
    print(segment_summary)
    
    return regional_summary, segment_summary

def analyze_products(df):
    """Analyze product performance"""
    print("\n" + "="*50)
    print("PRODUCT PERFORMANCE ANALYSIS")
    print("="*50)
    
    # Product performance by revenue
    product_revenue = df.groupby(['Product Name', 'Category']).agg({
        'Total Revenue': 'sum',
        'Total Sales': 'sum',
        'Quantity': 'sum',
        'Profit Margin': 'mean'
    }).round(2)
    
    # Top 5 best-selling products by revenue
    top_products = product_revenue.sort_values('Total Revenue', ascending=False).head(5)
    print("TOP 5 BEST-SELLING PRODUCTS BY REVENUE:")
    print(top_products)
    
    # Bottom 5 underperforming products by revenue
    bottom_products = product_revenue.sort_values('Total Revenue', ascending=True).head(5)
    print("\nTOP 5 UNDERPERFORMING PRODUCTS BY REVENUE:")
    print(bottom_products)
    
    return top_products, bottom_products, product_revenue

def analyze_categories_margins(df):
    """Analyze product categories by profit margins"""
    print("\n" + "="*50)
    print("CATEGORY MARGIN ANALYSIS")
    print("="*50)
    
    category_margins = df.groupby('Category').agg({
        'Profit Margin': 'mean',
        'Total Revenue': 'sum',
        'Total Sales': 'sum',
        'Order ID': 'count'
    }).round(2)
    category_margins = category_margins.sort_values('Profit Margin', ascending=False)
    
    print("CATEGORIES BY PROFIT MARGIN:")
    print(category_margins)
    
    return category_margins

def analyze_order_mode(df):
    """Analyze sales distribution by order mode"""
    print("\n" + "="*50)
    print("ORDER MODE ANALYSIS")
    print("="*50)
    
    order_mode_analysis = df.groupby('Order Mode').agg({
        'Total Sales': 'sum',
        'Total Revenue': 'sum',
        'Order ID': 'count',
        'Discount': 'mean'
    }).round(2)
    
    # Calculate percentages
    order_mode_analysis['Sales %'] = (order_mode_analysis['Total Sales'] / 
                                    order_mode_analysis['Total Sales'].sum() * 100).round(2)
    order_mode_analysis['Revenue %'] = (order_mode_analysis['Total Revenue'] / 
                                      order_mode_analysis['Total Revenue'].sum() * 100).round(2)
    
    print("ORDER MODE DISTRIBUTION:")
    print(order_mode_analysis)
    
    return order_mode_analysis

def create_visualizations(df, regional_summary, segment_summary, category_margins, order_mode_analysis):
    """Create comprehensive visualizations"""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('ACE Superstore Sales Analysis Dashboard', fontsize=20, y=0.98)
    
    # 1. Regional Revenue Performance
    regional_summary.sort_values('Total Revenue', ascending=True).plot(
        y='Total Revenue', kind='barh', ax=axes[0,0], color='skyblue'
    )
    axes[0,0].set_title('Total Revenue by Region', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Total Revenue ($)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Segment Performance
    segment_summary.plot(y='Total Revenue', kind='bar', ax=axes[0,1], color='lightcoral')
    axes[0,1].set_title('Total Revenue by Segment', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Total Revenue ($)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Order Mode Distribution
    order_mode_data = order_mode_analysis['Total Sales']
    axes[1,0].pie(order_mode_data.values, labels=order_mode_data.index, autopct='%1.1f%%',
                  colors=['lightgreen', 'gold'])
    axes[1,0].set_title('Sales Distribution by Order Mode', fontsize=14, fontweight='bold')
    
    # 4. Top Categories by Profit Margin
    top_margin_categories = category_margins.head(8)
    top_margin_categories.plot(y='Profit Margin', kind='bar', ax=axes[1,1], color='mediumpurple')
    axes[1,1].set_title('Top Categories by Profit Margin', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Profit Margin (%)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('ace_superstore_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional interactive visualizations with Plotly
    create_interactive_visualizations(df, regional_summary, segment_summary)

def create_interactive_visualizations(df, regional_summary, segment_summary):
    """Create interactive visualizations using Plotly"""
    
    # 1. Regional Performance Heatmap
    fig1 = px.bar(regional_summary.reset_index(), 
                  x='Region_Final', y='Total Revenue',
                  title='Regional Revenue Performance',
                  color='Total Revenue',
                  color_continuous_scale='Blues')
    fig1.update_layout(xaxis_tickangle=-45)
    fig1.write_html('regional_performance.html')
    
    # 2. Monthly Sales Trend
    monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Total Sales'].sum()
    fig2 = px.line(x=monthly_sales.index.astype(str), y=monthly_sales.values,
                   title='Monthly Sales Trend',
                   labels={'x': 'Month', 'y': 'Total Sales ($)'})
    fig2.write_html('monthly_sales_trend.html')
    
    # 3. Product Category Revenue Sunburst
    category_revenue = df.groupby(['Category', 'Sub-Category'])['Total Revenue'].sum().reset_index()
    fig3 = px.sunburst(category_revenue, path=['Category', 'Sub-Category'], 
                       values='Total Revenue',
                       title='Product Category Revenue Breakdown')
    fig3.write_html('category_revenue_sunburst.html')
    
    print("Interactive visualizations saved as HTML files")

def generate_insights_report(df, regional_summary, segment_summary, top_products, bottom_products, 
                           category_margins, order_mode_analysis):
    """Generate comprehensive insights and recommendations"""
    
    report = f"""
# ACE SUPERSTORE SALES ANALYSIS REPORT
**Business Intelligence Report**
**Date: July 4, 2025**
**Prepared by: Data Analytics Team**

---

## EXECUTIVE SUMMARY

ACE Superstore has demonstrated strong performance across multiple regions and product segments. 
This analysis covers {len(df):,} orders from {df['Order Date'].min().strftime('%B %Y')} to {df['Order Date'].max().strftime('%B %Y')}, 
generating **${df['Total Revenue'].sum():,.2f}** in total revenue.

### Key Metrics:
- **Total Orders**: {len(df):,}
- **Total Revenue**: ${df['Total Revenue'].sum():,.2f}
- **Total Sales**: ${df['Total Sales'].sum():,.2f}
- **Average Order Value**: ${df['Total Sales'].mean():.2f}
- **Unique Customers**: {df['Customer ID'].nunique():,}
- **Product Portfolio**: {df['Product ID'].nunique():,} products

---

## REGIONAL PERFORMANCE ANALYSIS

### Top Performing Regions:
"""
    
    # Add regional insights
    for region in regional_summary.head(3).index:
        revenue = regional_summary.loc[region, 'Total Revenue']
        orders = regional_summary.loc[region, 'Order Count']
        avg_discount = regional_summary.loc[region, 'Avg Discount Rate']
        report += f"\n**{region}**: ${revenue:,.2f} revenue ({orders:,} orders, {avg_discount:.1%} avg discount)"
    
    report += f"""

### Regional Insights:
- **{regional_summary.index[0]}** leads in revenue generation with ${regional_summary.iloc[0]['Total Revenue']:,.2f}
- Regional discount rates range from {regional_summary['Avg Discount Rate'].min():.1%} to {regional_summary['Avg Discount Rate'].max():.1%}
- Total regional coverage: {len(regional_summary)} regions

---

## SEGMENT PERFORMANCE ANALYSIS

### Top Performing Segments:
"""
    
    # Add segment insights
    for segment in segment_summary.head(3).index:
        revenue = segment_summary.loc[segment, 'Total Revenue']
        orders = segment_summary.loc[segment, 'Order Count']
        report += f"\n**{segment}**: ${revenue:,.2f} revenue ({orders:,} orders)"
    
    report += f"""

### Segment Insights:
- **{segment_summary.index[0]}** dominates with ${segment_summary.iloc[0]['Total Revenue']:,.2f} in revenue
- Segment distribution shows balanced portfolio across {len(segment_summary)} major categories

---

## PRODUCT PERFORMANCE ANALYSIS

### Top 5 Best-Selling Products by Revenue:
"""
    
    for i, (product, data) in enumerate(top_products.iterrows(), 1):
        report += f"\n{i}. **{product[0]}** ({product[1]}): ${data['Total Revenue']:,.2f}"
    
    report += f"""

### Top 5 Underperforming Products by Revenue:
"""
    
    for i, (product, data) in enumerate(bottom_products.iterrows(), 1):
        report += f"\n{i}. **{product[0]}** ({product[1]}): ${data['Total Revenue']:,.2f}"
    
    report += f"""

---

## CATEGORY MARGIN ANALYSIS

### Highest Margin Categories:
"""
    
    for category in category_margins.head(5).index:
        margin = category_margins.loc[category, 'Profit Margin']
        revenue = category_margins.loc[category, 'Total Revenue']
        report += f"\n**{category}**: {margin:.1f}% margin (${revenue:,.2f} revenue)"
    
    report += f"""

### Margin Insights:
- **{category_margins.index[0]}** shows highest profitability at {category_margins.iloc[0]['Profit Margin']:.1f}%
- Average profit margin across all categories: {df['Profit Margin'].mean():.1f}%
- Margin optimization opportunities exist in lower-performing categories

---

## ORDER MODE ANALYSIS

### Channel Performance:
"""
    
    for mode in order_mode_analysis.index:
        sales = order_mode_analysis.loc[mode, 'Total Sales']
        percentage = order_mode_analysis.loc[mode, 'Sales %']
        orders = order_mode_analysis.loc[mode, 'Order ID']
        report += f"\n**{mode}**: ${sales:,.2f} ({percentage:.1f}% of total sales, {orders:,} orders)"
    
    online_vs_store = order_mode_analysis.loc['Online', 'Sales %'] - order_mode_analysis.loc['In-Store', 'Sales %']
    leader = "Online" if online_vs_store > 0 else "In-Store"
    
    report += f"""

### Channel Insights:
- **{leader}** channel leads in sales performance
- Channel distribution shows {"balanced" if abs(online_vs_store) < 10 else "significant bias toward " + leader} performance
- Average discount rates: Online {order_mode_analysis.loc['Online', 'Discount']:.1%}, In-Store {order_mode_analysis.loc['In-Store', 'Discount']:.1%}

---

## KEY RECOMMENDATIONS

### 1. Regional Expansion Strategy
- **Focus on high-performing regions**: {regional_summary.index[0]} and {regional_summary.index[1]} show strong potential
- **Investigate underperforming regions**: Analyze market conditions in {regional_summary.index[-1]}
- **Standardize discount strategies**: Address regional discount rate variations

### 2. Product Portfolio Optimization
- **Promote high-margin categories**: Focus marketing on {category_margins.index[0]} and {category_margins.index[1]}
- **Review underperforming products**: Consider discontinuing or repositioning bottom revenue products
- **Expand successful product lines**: Increase inventory for top-performing products

### 3. Channel Strategy Enhancement
- **Optimize {leader.lower()} channel**: Leverage the stronger channel for growth
- **Improve omnichannel experience**: Balance online and in-store performance
- **Personalize discount strategies**: Tailor promotions by channel and customer segment

### 4. Profitability Improvement
- **Margin enhancement**: Focus on categories with profit margins above {df['Profit Margin'].mean():.1f}%
- **Cost optimization**: Review cost structures for low-margin categories
- **Premium positioning**: Develop premium product lines in high-margin categories

---

## CONCLUSION

ACE Superstore demonstrates solid performance across regions and segments. The analysis reveals opportunities for:
- Regional expansion in high-performing markets
- Product portfolio optimization focusing on high-margin categories
- Channel strategy enhancement leveraging strongest performing modes
- Targeted promotional strategies based on regional and segment insights

**Next Steps**: Implement quarterly review cycles to monitor KPIs and adjust strategies based on market dynamics.

---

*This report provides foundational insights for executive decision-making and strategic planning initiatives.*
"""
    
    return report

def main():
    """Main execution function"""
    print("="*60)
    print("ACE SUPERSTORE SALES ANALYSIS")
    print("="*60)
    
    # Load data
    sales_df, stores_df = load_data()
    
    # Clean and prepare data
    df = clean_and_prepare_data(sales_df, stores_df)
    
    # Perform analyses
    df = exploratory_analysis(df)
    regional_summary, segment_summary = analyze_by_region_segment(df)
    top_products, bottom_products, product_revenue = analyze_products(df)
    category_margins = analyze_categories_margins(df)
    order_mode_analysis = analyze_order_mode(df)
    
    # Create visualizations
    create_visualizations(df, regional_summary, segment_summary, category_margins, order_mode_analysis)
    
    # Generate comprehensive report
    report = generate_insights_report(df, regional_summary, segment_summary, top_products, 
                                    bottom_products, category_margins, order_mode_analysis)
    
    # Save report
    with open('ACE_Superstore_Analysis_Report.md', 'w') as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Files generated:")
    print("- ACE_Superstore_Analysis_Report.md")
    print("- ace_superstore_analysis_dashboard.png")
    print("- regional_performance.html")
    print("- monthly_sales_trend.html")
    print("- category_revenue_sunburst.html")
    print("="*60)

if __name__ == "__main__":
    main()
