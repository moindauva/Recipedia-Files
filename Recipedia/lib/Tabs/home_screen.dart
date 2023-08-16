// ignore_for_file: use_build_context_synchronously

import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:recipedia/Tabs/profile.dart';
import 'package:recipedia/Tabs/setting.dart';
import 'package:recipedia/WidgetsAndUtils/shared_preferences.dart';
import 'package:recipedia/WidgetsAndUtils/toast_message.dart';
import 'package:recipedia/api.dart';
import 'package:recipedia/scan_feature.dart';
import '../Admin/Recipes/recipe_card_slider.dart';
import '../Admin/Recipes/recipes.dart';
import '../Admin/recipes/category_tile.dart';
import 'package:dynamic_height_grid_view/dynamic_height_grid_view.dart';
import '../Admin/recipes/recipe_card.dart';
import '../Database/databaseHelper.dart';
import '../WidgetsAndUtils/automated_slider.dart';
import '../WidgetsAndUtils/progress_bar.dart';
import 'favorite.dart';
import 'dart:io';
import 'package:get/get.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({
    Key? key,
  }) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String name = 'Loading...';
  String profilePicture =
      'https://static.vecteezy.com/system/resources/thumbnails/005/544/718/small/profile-icon-design-free-vector.jpg';
  late String base64Image;
  late bool validIngredient;
  late String base64ImageTemp;
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

  late List<Map<String, dynamic>> recipesSlider;
  late List<Map<String, dynamic>> recipes = [];
  BuildContext? dialogContext;

  @override
  void initState() {
    super.initState();
    categoryValue = categories[0]['title'];
    loadData();
  }

  void loadData() async {
    String tempName = (await SharedPreference().getCred('name'))!;
    String tempURL = (await SharedPreference().getCred('profilePicture'))!;
    setState(() {
      name = tempName;
      profilePicture = tempURL;
    });
  }

  Future<List<Map<String, dynamic>>>
      getRecipesDataFromLocalDatabaseSlider() async {
    List<Map<String, dynamic>> recipesSlider =
        await DatabaseHelper().random5Recipes();
    //await Future.delayed(const Duration(seconds: 3));
    return recipesSlider;
  }

  Future<List<Map<String, dynamic>>> getRecipesDataFromLocalDatabase() async {
    List<Map<String, dynamic>> recipes =
        await DatabaseHelper().getAllRecipe('recipes', categoryValue);
    //await Future.delayed(const Duration(seconds: 3));
    return recipes;
  }

  String firstName(String str) {
    if (str.contains(' ')) {
      return str.substring(0, str.indexOf(' '));
    } else {
      return str;
    }
  }

  void getImageFromCamera() async {
    final picker = ImagePicker();
    final pickedFile = await picker.getImage(source: ImageSource.camera);

    WidgetsBinding.instance.addPostFrameCallback((_) async {
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (BuildContext context) {
          dialogContext = context;
          return ProgressBar(message: 'Loading...');
        },
      );
    });

    if (pickedFile != null) {
      setState(() {
        // Get the image as bytes
        File imageFile = File(pickedFile.path);
        List<int> imageBytes = imageFile.readAsBytesSync();

        // Convert image to base64
        base64ImageTemp = base64Encode(imageBytes);
      });
      base64Image = ',$base64ImageTemp';

      // Pass the base64 image to another function
      List itemList = await API(base64Image, context);
      if (itemList == null || itemList.isEmpty){
        validIngredient = false;
      }
      else{
        validIngredient = true;
      }

      Navigator.pop(dialogContext!);

      Navigator.push(
        context,
        PageRouteBuilder(
          transitionDuration: const Duration(milliseconds: 250),
          transitionsBuilder:
              (context, animation, secondaryAnimation, child) {
            return SlideTransition(
              position: Tween<Offset>(
                begin: const Offset(-1.0, 0.0),
                end: Offset.zero,
              ).animate(animation),
              child: child,
            );
          },
          pageBuilder: (context, animation, animationTime) {
            return ScanFeature(itemsScanned: itemList, validIngredient: validIngredient,);
          },
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    final currentTime = DateTime.now();
    final currentHour = currentTime.hour;
    String mealTime;

    if (currentHour >= 6 && currentHour < 12) {
      mealTime = 'Breakfast';
    } else if (currentHour >= 12 && currentHour < 18) {
      mealTime = 'Lunch';
    } else {
      mealTime = 'Dinner';
    }

    return Scaffold(
      backgroundColor: const Color(0XFFFBFBFB),
      //backgroundColor: Color(0XFF212121),
      resizeToAvoidBottomInset: true,
      floatingActionButton: FloatingActionButton(
        backgroundColor: Colors.black87,
        onPressed: () async {
          getImageFromCamera();
        },
        child: const Icon(
          Icons.camera_alt,
          color: Colors.white,
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      bottomNavigationBar: SafeArea(
        child: BottomAppBar(
          elevation: 0,
          color: Colors.transparent,
          shape: const CircularNotchedRectangle(),
          notchMargin: 5,
          clipBehavior: Clip.hardEdge,
          child: Container(
            height: kToolbarHeight,
            color: const Color(0XFFFF4F5A),
            child: Padding(
              padding: const EdgeInsets.only(left: 25, right: 25),
              child: Row(
                mainAxisSize: MainAxisSize.max,
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: <Widget>[
                  IconButton(
                    icon: const Icon(
                      Icons.home,
                      color: Colors.white,
                    ),
                    onPressed: () {},
                  ),
                  //Favorite Button
                  IconButton(
                    icon: const Icon(
                      Icons.favorite_rounded,
                      color: Colors.white,
                    ),
                    onPressed: () {
                      Navigator.pushAndRemoveUntil(
                        context,
                        PageRouteBuilder(
                          transitionDuration: const Duration(milliseconds: 250),
                          transitionsBuilder:
                              (context, animation, secondaryAnimation, child) {
                            return SlideTransition(
                              position: Tween<Offset>(
                                begin: const Offset(-1.0, 0.0),
                                end: Offset.zero,
                              ).animate(animation),
                              child: child,
                            );
                          },
                          pageBuilder: (context, animation, animationTime) {
                            return const Favorite();
                          },
                        ),
                        (route) => false,
                      );
                    },
                  ),
                  const Padding(
                    padding: EdgeInsets.only(left: 25, right: 25),
                    child: SizedBox(),
                  ),
                  //Profile Button
                  IconButton(
                    icon: const Icon(
                      Icons.person,
                      color: Colors.white,
                    ),
                    onPressed: () {
                      Navigator.pushAndRemoveUntil(
                        context,
                        PageRouteBuilder(
                          transitionDuration: const Duration(milliseconds: 250),
                          transitionsBuilder:
                              (context, animation, secondaryAnimation, child) {
                            return SlideTransition(
                              position: Tween<Offset>(
                                begin: const Offset(-1.0, 0.0),
                                end: Offset.zero,
                              ).animate(animation),
                              child: child,
                            );
                          },
                          pageBuilder: (context, animation, animationTime) {
                            return const Profile();
                          },
                        ),
                        (route) => false,
                      );
                    },
                  ),
                  IconButton(
                    icon: const Icon(
                      Icons.settings,
                      color: Colors.white,
                    ),
                    onPressed: () {
                      Navigator.pushAndRemoveUntil(
                        context,
                        PageRouteBuilder(
                          transitionDuration: const Duration(milliseconds: 250),
                          transitionsBuilder:
                              (context, animation, secondaryAnimation, child) {
                            return SlideTransition(
                              position: Tween<Offset>(
                                begin: const Offset(-1.0, 0.0),
                                end: Offset.zero,
                              ).animate(animation),
                              child: child,
                            );
                          },
                          pageBuilder: (context, animation, animationTime) {
                            return const Setting();
                          },
                        ),
                        (route) => false,
                      );
                    },
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
      body: SingleChildScrollView(
        scrollDirection: Axis.vertical,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            //User's name and picture
            Container(
              padding: const EdgeInsets.only(
                left: 25.0,
                right: 25.0,
                top: 40,
                bottom: 25.0,
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Column(
                    mainAxisAlignment: MainAxisAlignment.start,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Hi, ${firstName(name)}',
                        style: const TextStyle(
                          color: Colors.black,
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        'Ready to cook for $mealTime?',
                        style: const TextStyle(
                          color: Colors.black54,
                          fontSize: 17,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  Container(
                    width: 55.0,
                    height: 55.0,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(20.0),
                      color: Colors.grey[300],
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(20.0),
                      child: Image.network(
                        profilePicture,
                        fit: BoxFit.cover,
                      ),
                    ),
                  )
                ],
              ),
            ),
            //Recipes Slider
            FutureBuilder<List<Map<String, dynamic>>>(
              future: getRecipesDataFromLocalDatabaseSlider(),
              builder: (BuildContext context,
                  AsyncSnapshot<List<Map<String, dynamic>>> snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return SizedBox(
                    height: size.height * 0.20,
                    child: const Center(
                      child: CircularProgressIndicator(),
                    ),
                  );
                } else if (snapshot.hasError) {
                  return Text('Error: ${snapshot.error}');
                } else {
                  List<Map<String, dynamic>> recipesSlider = snapshot.data!;
                  return AutomatedSlider(
                    itemCount: 5,
                    items: List.generate(
                      recipesSlider.length,
                      (index) => RecipeCardSlider(
                        isAdmin: false,
                        id: (recipesSlider[index]['recipeId']).toString(),
                        title: recipesSlider[index]['recipeName'],
                        description: recipesSlider[index]['recipeDescription'],
                        course: recipesSlider[index]['recipeCourse'],
                        diet: recipesSlider[index]['recipeDiet'],
                        cookTime: recipesSlider[index]['recipeTime'],
                        rating: recipesSlider[index]['recipeRating'],
                        thumbnailUrl: recipesSlider[index]['recipeURL'],
                        ingredients: recipesSlider[index]['recipeIngredients'],
                        category: recipesSlider[index]['recipeCategory'],
                      ),
                    ),
                  );
                }
              },
            ),
            SizedBox(
              height: size.height * 0.03,
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
            Container(
              padding: const EdgeInsets.symmetric(
                horizontal: 10,
              ),
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
                    height: size.height * 0.75,
                    child: CustomScrollView(
                      slivers: [
                        SliverDynamicHeightGridView(
                          itemCount: recipes.length,
                          crossAxisCount: 2,
                          builder: (ctx, index) {
                            return RecipeCard(
                              isAdmin: false,
                              id: (recipes[index]['recipeId']).toString(),
                              title: recipes[index]['recipeName'],
                              description: recipes[index]['recipeDescription'],
                              course: recipes[index]['recipeCourse'],
                              diet: recipes[index]['recipeDiet'],
                              cookTime: recipes[index]['recipeTime'],
                              rating: recipes[index]['recipeRating'],
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
