package org.tetrabox.minijava.xtext.typing

import miniJava.Assignment
import miniJava.BoolConstant
import miniJava.BooleanTypeRef
import miniJava.Clazz
import miniJava.ClassRef
import miniJava.Expression
import miniJava.IntConstant
import miniJava.IntegerTypeRef
import miniJava.Method
import miniJava.MiniJavaFactory
import miniJava.MiniJavaPackage
import miniJava.Null
import miniJava.Return
import miniJava.StringConstant
import miniJava.StringTypeRef
import miniJava.Super
import miniJava.SymbolRef
import miniJava.This
import miniJava.TypeRef
import miniJava.VariableDeclaration

import static extension org.eclipse.xtext.EcoreUtil2.*
import miniJava.FieldAccess
import miniJava.MethodCall
import miniJava.VoidTypeRef
import miniJava.TypeDeclaration
import miniJava.Interface
import miniJava.TypedDeclaration
import miniJava.NewObject

class MiniJavaTypeComputer {
	private static val factory = MiniJavaFactory.eINSTANCE
	public static val STRING_TYPE = factory.createClazz => [name = 'stringType']
	public static val INT_TYPE = factory.createClazz => [name = 'intType']
	public static val BOOLEAN_TYPE = factory.createClazz => [name = 'booleanType']
	public static val NULL_TYPE = factory.createClazz => [name = 'nullType']

	static val ep = MiniJavaPackage.eINSTANCE

	def TypeDeclaration getType(TypeRef r) {
		switch r {
			ClassRef: r.referencedClass
			IntegerTypeRef: INT_TYPE
			BooleanTypeRef: BOOLEAN_TYPE
			StringTypeRef: STRING_TYPE
			VoidTypeRef : NULL_TYPE
		}
	}

	def TypeDeclaration typeFor(Expression e) {
		switch (e) {
			SymbolRef:
				e.symbol.typeRef.type
			FieldAccess:
				e.field.typeRef.type
			MethodCall:
				e.method.typeRef.type
			NewObject:
				e.type
			This:
				e.getContainerOfType(Clazz)
			Super:
				e.getContainerOfType(Clazz).superClass
			Null:
				NULL_TYPE
			StringConstant:
				STRING_TYPE
			IntConstant:
				INT_TYPE
			BoolConstant:
				BOOLEAN_TYPE
		}
	}


	def isPrimitive(TypeDeclaration c) {
		c.eResource === null
	}

	def TypeDeclaration expectedType(Expression e) {
		val c = e.eContainer
		val f = e.eContainingFeature
		switch (c) {
			VariableDeclaration:
				c.typeRef.type
			Assignment case f == ep.assignment_Value: {
				val assignee = c.assignee
				switch (assignee) {
					VariableDeclaration: assignee.typeRef.type
					FieldAccess: assignee.typeFor
				}
			}
			Return:
				c.getContainerOfType(Method).typeRef.type
			case f == ep.ifStatement_Expression:
				BOOLEAN_TYPE
			MethodCall case f == ep.methodCall_Args: {
				if (c.method !== null) {
				if (c.method.params.size > c.args.indexOf(e))
					c.method.params.get(c.args.indexOf(e)).typeRef.type
				}
			}
			NewObject case f == ep.newObject_Args: {
				c.type.members.filter(Method).findFirst[it.name === null && it.params.size === c.args.size].params.get(c.args.indexOf(e)).typeRef.type
			}
		}
	}
}
