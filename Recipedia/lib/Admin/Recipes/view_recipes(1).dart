import 'package:dynamic_height_grid_view/dynamic_height_grid_view.dart';
import 'package:flutter/material.dart';
import 'package:recipedia/Admin/recipes/recipe_card.dart';

import '../../Database/databaseHelper.dart';
import 'category_tile.dart';

class ViewRecipes extends StatefulWidget {
  const ViewRecipes({
    Key? key,
  }) : super(key: key);

  @override
  State<ViewRecipes> createState() => _ViewRecipesState();
}

class _ViewRecipesState extends State<ViewRecipes> {
  String categoryValue = 'All';
  List<Map<String, dynamic>> categories = [
    {
      'image': 'assets/all.png',
      'title': 'All',
    },
    {
      'image': 'assets/appetizer.png',
      'title': 'Appetizer',
    },
    {
      'image': 'assets/main-course.png',
      'title': 'Main Course',
    },
    {
      'image': 'assets/breakfast.png',
      'title': 'Breakfast',
    },
    {
      'image': 'assets/lunch.png',
      'title': 'Lunch',
    },
    {
      'image': 'assets/dinner.png',
      'title': 'Dinner',
    },
    {
      'image': 'assets/side-dish.png',
      'title': 'Side Dish',
    },
    {
      'image': 'assets/dessert.png',
      'title': 'Dessert',
    },
    {
      'image': 'assets/bruntch.png',
      'title': 'Bruntch',
    },
    {
      'image': 'assets/snack.png',
      'title': 'Snack',
    },
    {
      'image': 'assets/one-pot-dish.png',
      'title': 'One Pot',
    }
  ];

  late List<Map<String, dynamic>> recipes;

  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    loadData();
  }

  Future<void> loadData() async {
    recipes = await getRecipesDataFromLocalDatabase();

    await Future.delayed(const Duration(seconds: 3));
    setState(() {
      isLoading = false;
    });
  }

  Future<List<Map<String, dynamic>>> getRecipesDataFromLocalDatabase() async {
    List<Map<String, dynamic>> recipes =
        await DatabaseHelper().getAllRecipe('recipes', categoryValue);
    await Future.delayed(const Duration(seconds: 3));
    return recipes;
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Scaffold(
      backgroundColor: const Color(0XFFFBFBFB),
      body: SingleChildScrollView(
        scrollDirection: Axis.vertical,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            SizedBox(
              height: size.height * 0.05,
            ),
            //Categories Heading
            Container(
              alignment: Alignment.centerLeft,
              padding: const EdgeInsets.only(
                left: 25.0,
                right: 25.0,
              ),
              child: const Text(
                'Categories',
                style: TextStyle(
                  fontSize: 24.0,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            SizedBox(
              height: size.height * 0.015,
            ),
            //Horizontal Category Tiles
            SizedBox(
              height: 70,
              child: ListView(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.only(top: 0),
                children: categories
                    .map(
                      (category) => GestureDetector(
                        onTap: () {
                          setState(() {
                            categoryValue = category['title'];
                          });
                        },
                        child: CategoryTile(
                          text: category['title'],
                          image: category['image'],
                          selected: category['title'] == categoryValue,
                        ),
                      ),
                    )
                    .toList(),
              ),
              /*ListView.builder(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.only(top: 0),
                itemCount: categories.length,
                itemBuilder: (context, index) {
                  return CategoryTile(
                    image: categories[index]['image'],
                    text: categories[index]['title'],
                  );
                },
              ),*/
            ),
            SizedBox(
              height: size.height * 0.015,
            ),
            //Recipes Grid
            FutureBuilder<List<Map<String, dynamic>>>(
              future: getRecipesDataFromLocalDatabase(),
              builder: (BuildContext context,
                  AsyncSnapshot<List<Map<String, dynamic>>> snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return SizedBox(
                    height: size.height * 0.5,
                    child: const Center(
                      child: CircularProgressIndicator(),
                    ),
                  );
                } else if (snapshot.hasError) {
                  return Text('Error: ${snapshot.error}');
                } else {
                  List<Map<String, dynamic>> recipes = snapshot.data!;
                  return SizedBox(
                    height: size.height * 0.8,
                    child: CustomScrollView(
                      slivers: [
                        SliverDynamicHeightGridView(
                          itemCount: recipes.length,
                          crossAxisCount: 2,
                          builder: (ctx, index) {
                            return RecipeCard(
                              isAdmin: true,
                              id: (recipes[index]['recipeId']).toString(),
                              title: recipes[index]['recipeName'],
                              description: recipes[index]['recipeDescription'],
                              course: recipes[index]['recipeCourse'],
                              diet: recipes[index]['recipeDiet'],
                              cookTime: recipes[index]['recipeTime'],
                              rating:
                                  recipes[index]['recipeRating'],
                              thumbnailUrl: recipes[index]['recipeURL'],
                              ingredients: recipes[index]['recipeIngredients'],
                              category: recipes[index]['recipeCategory'],
                            );
                          },
                        ),
                      ],
                    ),
                  );
                }
              },
            ),
          ],
        ),
      ),
    );
  }
}
