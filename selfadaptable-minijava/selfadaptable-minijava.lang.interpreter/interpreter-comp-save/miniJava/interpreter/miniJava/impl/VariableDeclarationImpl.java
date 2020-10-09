package miniJava.interpreter.miniJava.impl;

import java.lang.Override;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.VariableDeclaration;
import org.eclipse.emf.ecore.EClass;

public class VariableDeclarationImpl extends SymbolImpl implements VariableDeclaration {
	protected VariableDeclarationImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.VARIABLE_DECLARATION;
	}
}
