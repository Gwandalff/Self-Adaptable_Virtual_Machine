package miniJava.interpreter.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import java.util.Collection;
import miniJava.interpreter.miniJava.Block;
import miniJava.interpreter.miniJava.Clazz;
import miniJava.interpreter.miniJava.Context;
import miniJava.interpreter.miniJava.Expression;
import miniJava.interpreter.miniJava.Field;
import miniJava.interpreter.miniJava.FieldBinding;
import miniJava.interpreter.miniJava.Member;
import miniJava.interpreter.miniJava.Method;
import miniJava.interpreter.miniJava.MiniJavaFactory;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.NewCall;
import miniJava.interpreter.miniJava.NewObject;
import miniJava.interpreter.miniJava.ObjectInstance;
import miniJava.interpreter.miniJava.ObjectRefValue;
import miniJava.interpreter.miniJava.Parameter;
import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.SymbolBinding;
import miniJava.interpreter.miniJava.Value;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;
import org.eclipse.emf.ecoretools.ale.compiler.lib.CollectionService;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;

public class NewObjectImpl extends ExpressionImpl implements NewObject {
	protected Clazz type;

	protected EList<Expression> args;

	protected NewObjectImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.NEW_OBJECT;
	}

	public Clazz getType() {
		if (type != null && type.eIsProxy()) {
			InternalEObject oldType = (InternalEObject) type;
			type = (Clazz) eResolveProxy(oldType);
			if (type != oldType) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, MiniJavaPackage.NEW_OBJECT__TYPE, oldType, type));
			}
		}
		return type;
	}

	public Clazz basicGetType() {
		return type;
	}

	public void setType(Clazz newType) {
		Clazz oldType = type;
		type = newType;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.NEW_OBJECT__TYPE, oldType, type));
	}

	public EList<Expression> getArgs() {
		if (args == null) {
			args = new EObjectContainmentEList<Expression>(Expression.class, this, MiniJavaPackage.NEW_OBJECT__ARGS);
		}
		return args;
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.NEW_OBJECT__ARGS :
				return ((InternalEList<?>) getArgs()).basicRemove(otherEnd, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.NEW_OBJECT__TYPE :
				if (resolve)
					return getType();
				return basicGetType();
			case MiniJavaPackage.NEW_OBJECT__ARGS :
				return getArgs();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.NEW_OBJECT__TYPE :
				setType((Clazz) newValue);
				return;
			case MiniJavaPackage.NEW_OBJECT__ARGS :
				getArgs().clear();
				getArgs().addAll((Collection<? extends Expression>) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.NEW_OBJECT__TYPE :
				setType((Clazz) null);
				return;
			case MiniJavaPackage.NEW_OBJECT__ARGS :
				getArgs().clear();
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.NEW_OBJECT__TYPE :
				return type != null;
			case MiniJavaPackage.NEW_OBJECT__ARGS :
				return args != null && !args.isEmpty();
		}
		return super.eIsSet(featureID);
	}

	public Value evaluateExpression(State state) {
		Value result;
		ObjectInstance res = ((ObjectInstance) (MiniJavaFactory.eINSTANCE.createObjectInstance()));
		res.setType(this.getType());
		state.getObjectsHeap().add(res);
		int i = ((int) (0));
		int z = ((int) (CollectionService.size(res.getType().getMembers())));
		while ((i) < (z)) {
			Member m = ((Member) (CollectionService.get(res.getType().getMembers(), i)));
			if (m instanceof Field) {
				Field f = ((Field) (m));
				if (!EqualService.equals((f.getDefaultValue()), (null))) {
					Value v = ((Value) (((Expression) (f.getDefaultValue())).evaluateExpression((State) (state))));
					FieldBinding fb = ((FieldBinding) (MiniJavaFactory.eINSTANCE.createFieldBinding()));
					fb.setField(f);
					fb.setValue(v);
					res.getFieldbindings().add(fb);
				}
			}
			i = (i) + (1);
		}
		i = 0;
		Method constructor = ((Method) (null));
		while ((((i) < (z)) && (EqualService.equals((constructor), (null))))) {
			Member m = ((Member) (CollectionService.get(res.getType().getMembers(), i)));
			if (m instanceof Method) {
				Method mtd = ((Method) (m));
				if (((EqualService.equals((mtd.getName()), (null))) && (EqualService.equals((CollectionService.size(mtd.getParams())), (CollectionService.size(this.getArgs())))))) {
					constructor = mtd;
				}
			}
			i = (i) + (1);
		}
		if (!EqualService.equals((constructor), (null))) {
			Context newContext = ((Context) (MiniJavaFactory.eINSTANCE.createContext()));
			i = 0;
			z = CollectionService.size(this.getArgs());
			while ((i) < (z)) {
				Expression arg = ((Expression) (CollectionService.get(this.getArgs(), i)));
				Parameter param = ((Parameter) (CollectionService.get(constructor.getParams(), i)));
				SymbolBinding binding = ((SymbolBinding) (MiniJavaFactory.eINSTANCE.createSymbolBinding()));
				binding.setSymbol(param);
				binding.setValue(((Expression) (arg)).evaluateExpression((State) (state)));
				i = (i) + (1);
				newContext.getBindings().add(binding);
			}
			NewCall call = ((NewCall) (MiniJavaFactory.eINSTANCE.createNewCall()));
			call.setNewz(this);
			((State) (state)).pushNewFrame((ObjectInstance) (res), (NewCall) (call), (Context) (newContext));
			Block bd = ((Block) (constructor.getBody()));
			((Block) (bd)).evaluateStatement((State) (state));
			((State) (state)).popCurrentFrame();
		}
		ObjectRefValue tmp = ((ObjectRefValue) (MiniJavaFactory.eINSTANCE.createObjectRefValue()));
		tmp.setInstance(res);
		result = (Value) (tmp) ;
		return result;
	}
}
